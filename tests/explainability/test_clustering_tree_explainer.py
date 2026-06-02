import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as Sklearn_KMeans
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml import Pipeline

from mercury.explainability.explainers.clustering_tree_explainer import ClusteringTreeExplainer

pytestmark = pytest.mark.usefixtures("model_and_data_cte")


@pytest.fixture(scope="module")
def model_and_data_cte():
    k = 3
    random_state = 42
    x_data, _ = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=k,
        cluster_std=2.5,
        random_state=random_state,
    )

    feature_names = ["feature_1", "feature_2"]
    pandas_df = pd.DataFrame(x_data, columns=feature_names)

    sk_kmeans = Sklearn_KMeans(k, random_state=random_state)
    sk_kmeans.fit(x_data)

    sk_pipeline = SklearnPipeline(
        steps=[("scaler", StandardScaler()), ("kmeans", Sklearn_KMeans(k, random_state=random_state))]
    )
    sk_pipeline.fit(x_data)

    spark_sess = None
    spark_df = None
    spark_pipeline_model = None
    spark_df_2 = None
    spark_kmeans_model = None

    try:
        spark_sess = SparkSession.builder.master("local[2]").appName("cte-tests").getOrCreate()
        spark_df = spark_sess.createDataFrame(pandas_df)
        assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
        spark_kmeans = SparkKMeans(k=k, seed=random_state)
        spark_pipeline = Pipeline(stages=[assembler, spark_kmeans])
        spark_pipeline_model = spark_pipeline.fit(spark_df)

        spark_df_2 = assembler.transform(spark_df).select("features")
        spark_kmeans_model = spark_kmeans.fit(spark_df_2)
    except Exception:
        pass

    return {
        "spark_sess": spark_sess,
        "pandas_df": pandas_df,
        "sk_kmeans": sk_kmeans,
        "sk_pipeline": sk_pipeline,
        "spark_df": spark_df,
        "spark_pipeline_model": spark_pipeline_model,
        "spark_df_2": spark_df_2,
        "spark_kmeans_model": spark_kmeans_model,
        "K": k,
    }


def _require_spark(model_and_data_cte):
    if model_and_data_cte["spark_sess"] is None:
        pytest.skip("PySpark clustering tree tests require a working Spark/Java runtime")


def _assert_all_clusters_have_leafs(explanation):
    assert """label="0""" in str(explanation)
    assert """label="1""" in str(explanation)
    assert """label="2""" in str(explanation)


def test_clustering_tree_explainer_pandas(model_and_data_cte):
    sk_kmeans = model_and_data_cte['sk_kmeans']
    K = model_and_data_cte['K']
    pandas_df = model_and_data_cte['pandas_df']

    clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=sk_kmeans, max_leaves=K)
    explanation = clustering_tree_explainer.explain(pandas_df)
    plot_explanation = explanation.plot(filename="test1")

    _assert_all_clusters_have_leafs(str(plot_explanation))
    assert clustering_tree_explainer.tree._size() >= K
    assert clustering_tree_explainer.tree._max_depth() > 1

    score = clustering_tree_explainer.score(pandas_df)
    surrogate_score = clustering_tree_explainer.surrogate_score(pandas_df)
    assert isinstance(score, float) and isinstance(surrogate_score, float)
    assert surrogate_score >= score


def test_clustering_tree_explainer_pandas_sk_pipeline(model_and_data_cte):
    sk_pipeline = model_and_data_cte['sk_pipeline']
    K = model_and_data_cte['K']
    pandas_df = model_and_data_cte['pandas_df']

    clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=sk_pipeline, max_leaves=K)
    explanation = clustering_tree_explainer.explain(pandas_df)
    plot_explanation = explanation.plot(filename="test1")

    _assert_all_clusters_have_leafs(str(plot_explanation))
    assert clustering_tree_explainer.tree._size() >= K
    assert clustering_tree_explainer.tree._max_depth() > 1


def test_clustering_tree_explainer_scalers(model_and_data_cte):
    K = model_and_data_cte['K']
    pandas_df = model_and_data_cte['pandas_df'].copy()

    scalers = {}
    for c in pandas_df.columns:
        scalers[c] = StandardScaler()
        pandas_df[c] = scalers[c].fit_transform(pandas_df[c].values.reshape(-1, 1))

    sk_kmeans = Sklearn_KMeans(K, random_state=42)
    sk_kmeans.fit(pandas_df)

    clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=sk_kmeans, max_leaves=K)
    explanation = clustering_tree_explainer.explain(pandas_df)
    plot_explanation = explanation.plot(filename="test1", scalers=scalers)
    _assert_all_clusters_have_leafs(str(plot_explanation))


def test_clustering_tree_explainer_pandas_more_leaves(model_and_data_cte):
    """Test case when tree explainer grows to more leaves than clusters (ExKMC method)"""

    sk_kmeans = model_and_data_cte['sk_kmeans']
    K = model_and_data_cte['K']
    pandas_df = model_and_data_cte['pandas_df']

    clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=sk_kmeans, max_leaves=K+3)
    explanation = clustering_tree_explainer.explain(pandas_df)
    plot_explanation = explanation.plot(filename="test1")

    _assert_all_clusters_have_leafs(str(plot_explanation))
    # check that explanation has more nodes
    nodes = ["n_" + str(i) for i in range(7)]
    for n in nodes:
        assert n in str(plot_explanation)
    assert clustering_tree_explainer.tree._size() >= K

def test_clustering_tree_explainer_spark_pipeline(model_and_data_cte):
    _require_spark(model_and_data_cte)

    spark_pipeline_model = model_and_data_cte['spark_pipeline_model']
    K = model_and_data_cte['K']
    spark_df = model_and_data_cte['spark_df']

    clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=spark_pipeline_model, max_leaves=K)
    explanation = clustering_tree_explainer.explain(spark_df)
    plot_explanation = explanation.plot(filename="test1")
    _assert_all_clusters_have_leafs(str(plot_explanation))
    assert clustering_tree_explainer.tree._size() >= K


def test_clustering_tree_explainer_spark_kmeans(model_and_data_cte):
    _require_spark(model_and_data_cte)

    spark_kmeans_model = model_and_data_cte['spark_kmeans_model']
    K = model_and_data_cte['K']
    spark_df = model_and_data_cte['spark_df_2']

    clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=spark_kmeans_model, max_leaves=K)
    explanation = clustering_tree_explainer.explain(spark_df)
    plot_explanation = explanation.plot(filename="test1", feature_names=["feature_1", "feature_2"])

    _assert_all_clusters_have_leafs(str(plot_explanation))
    assert clustering_tree_explainer.tree._size() >= K


def test_clustering_tree_explainer_spark_subsampling(model_and_data_cte):
    _require_spark(model_and_data_cte)

    spark_pipeline_model = model_and_data_cte['spark_pipeline_model']
    K = model_and_data_cte['K']
    spark_df = model_and_data_cte['spark_df']

    clustering_tree_explainer = ClusteringTreeExplainer(
        clustering_model=spark_pipeline_model, max_leaves=K, verbose=True
    )
    explanation = clustering_tree_explainer.explain(spark_df, subsample=0.9)
    plot_explanation = explanation.plot(filename="test1")

    # check that any node has all the 1000 samples
    assert "samples=\1000" not in str(plot_explanation)


def test_clustering_tree_explainer_no_ibb(model_and_data_cte):
    sk_kmeans = model_and_data_cte['sk_kmeans']
    K = model_and_data_cte['K']
    pandas_df = model_and_data_cte['pandas_df']

    clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=sk_kmeans, max_leaves=K, base_tree='NONE')
    explanation = clustering_tree_explainer.explain(pandas_df)
    plot_explanation = explanation.plot(filename="test1")

    _assert_all_clusters_have_leafs(str(plot_explanation))


def test_clustering_tree_explainer_errors(model_and_data_cte):
    sk_kmeans = model_and_data_cte['sk_kmeans']
    K = model_and_data_cte['K']
    pandas_df = model_and_data_cte['pandas_df']

    with pytest.raises(Exception):
        clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=sk_kmeans, max_leaves=K-2)
        explanation = clustering_tree_explainer.explain(pandas_df)

    with pytest.raises(Exception):
        clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=sk_kmeans, max_leaves=K, base_tree="new_tree")
        explanation = clustering_tree_explainer.explain(pandas_df)


if __name__ == "__main__":
	pytest.main([__file__])
