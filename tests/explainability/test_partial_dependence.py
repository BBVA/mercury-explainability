
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mercury.explainability import PartialDependenceExplainer
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from pyspark.ml import Pipeline
import pyspark.ml.regression as pysparkreg
import pyspark.ml.classification as pysparkclas
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

import pytest

pytestmark = pytest.mark.usefixtures("model_and_data_pdp")


@pytest.fixture(scope="module")
def model_and_data_pdp():
    houses = fetch_california_housing()
    iris = load_iris()

    houses_pd_df = pd.DataFrame(houses['data'], columns=houses['feature_names'])
    houses_pd_df['target'] = houses['target']
    iris_pd_df = pd.DataFrame(iris['data'], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    iris_pd_df['target'] = iris['target']
    boston_pd_df = pd.read_csv("tests/explainability/model_and_data/boston.csv")

    rf_iris_sk = RandomForestClassifier().fit(
        iris_pd_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        iris_pd_df['target'],
    )
    rf_houses_sk = RandomForestRegressor().fit(
        houses_pd_df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']],
        houses_pd_df['target'],
    )
    rf_boston_sk = RandomForestRegressor().fit(
        boston_pd_df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']],
        boston_pd_df['target'],
    )

    spark_sess = None
    iris_sp_df = None
    houses_sp_df = None
    boston_sp_df = None
    rf_iris_sp = None
    assembler_iris = None
    rf_houses_sp = None
    assembler_houses = None
    assembler_boston = None
    rf_boston_sp = None

    try:
        spark_sess = SparkSession.builder.master("local[2]").appName("pdp-tests").getOrCreate()
        houses_sp_df = spark_sess.createDataFrame(houses_pd_df)
        iris_sp_df = spark_sess.createDataFrame(iris_pd_df)
        boston_sp_df = spark_sess.createDataFrame(boston_pd_df)

        assembler_iris = VectorAssembler(
            inputCols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            outputCol='features',
        )
        iris_sp_df_temp = assembler_iris.transform(iris_sp_df)
        rf_iris_sp = pysparkclas.RandomForestClassifier(featuresCol="features", labelCol="target").fit(iris_sp_df_temp)

        assembler_houses = VectorAssembler(
            inputCols=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'],
            outputCol='features',
        )
        houses_sp_df_temp = assembler_houses.transform(houses_sp_df)
        rf_houses_sp = pysparkreg.RandomForestRegressor(featuresCol="features", labelCol="target").fit(houses_sp_df_temp)

        assembler_boston = VectorAssembler(
            inputCols=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
            outputCol='features',
        )
        boston_sp_df_temp = assembler_boston.transform(boston_sp_df)
        rf_boston_sp = pysparkreg.RandomForestRegressor(featuresCol='features', labelCol='target').fit(boston_sp_df_temp)
    except Exception:
        pass

    return {
        'spark_sess': spark_sess,
        'iris_pd_df': iris_pd_df.loc[:, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        'houses_pd_df': houses_pd_df.loc[:, ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']],
        'boston_pd_df': boston_pd_df.loc[:, ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']],
        'iris_sp_df': iris_sp_df.drop("target") if iris_sp_df is not None else None,
        'houses_sp_df': houses_sp_df.drop("target") if houses_sp_df is not None else None,
        'boston_sp_df': boston_sp_df.drop("target") if boston_sp_df is not None else None,
        'rf_iris_sk': rf_iris_sk,
        'rf_houses_sk': rf_houses_sk,
        'rf_boston_sk': rf_boston_sk,
        'rf_iris_sp': rf_iris_sp,
        'assembler_iris': assembler_iris,
        'rf_houses_sp': rf_houses_sp,
        'assembler_houses': assembler_houses,
        'assembler_boston': assembler_boston,
        'rf_boston_sp': rf_boston_sp,
    }


def _require_spark(model_and_data_pdp):
    if model_and_data_pdp['spark_sess'] is None:
        pytest.skip("PySpark partial dependence tests require a working Spark/Java runtime")


def test_pandas_classification(model_and_data_pdp):
    """ Tests the explainer for  multinomial classification task using "plain Python"
        models
    """
    rf = model_and_data_pdp['rf_iris_sk']
    features = model_and_data_pdp['iris_pd_df']

    explainer = PartialDependenceExplainer(rf.predict_proba)
    explanation = explainer.explain(features)
    return_dict = explanation.data

    # Explanation should contain all the features
    assert list(return_dict.keys()) == list(features.columns)
    # Predictions should contain R^3 vectors
    assert return_dict[list(return_dict.keys())[0]]['preds'].shape[1] == 3
    # Quantiles should be R^3
    assert return_dict[list(return_dict.keys())[0]]['lower_quantile'].shape[1] == 3
    assert return_dict[list(return_dict.keys())[0]]['upper_quantile'].shape[1] == 3

    # # Check plotting doesn't crash
    # explanation.plot()


def test_pandas_regression(model_and_data_pdp):
    rf = model_and_data_pdp['rf_houses_sk']
    features = model_and_data_pdp['houses_pd_df']
    explainer = PartialDependenceExplainer(rf.predict, verbose=True)

    features_to_ignore = ['HouseAge', 'AveBedrms', 'Population']
    features_to_use = [f for f in list(features.columns) if f not in
            features_to_ignore]

    explanation = explainer.explain(features, ignore_feats=features_to_ignore)

    return_dict = explanation.data
    # Explanation should contain all the non-ignored features
    assert list(return_dict.keys()) == features_to_use
    # Predictions should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['preds'].shape) == 1
    # Quantiles should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['lower_quantile'].shape) == 1
    assert len(return_dict[list(return_dict.keys())[0]]['upper_quantile'].shape) == 1

    # # Check plotting doesn't crash
    # explanation.plot()


def test_pandas_regression_with_categoricals(model_and_data_pdp):
    rf = model_and_data_pdp['rf_boston_sk']
    features = model_and_data_pdp['boston_pd_df']
    explainer = PartialDependenceExplainer(rf.predict, quantiles=False)
    explanation = explainer.explain(features, ignore_feats=['PTRATIO', 'B', 'LSTAT', 'AGE'])
    return_dict = explanation.data
    # Predictions should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['preds'].shape) == 1

    assert len(return_dict['NOX']['values']) == 50

    # # Check plotting doesn't crash
    # explanation.plot()


def test_spark_classification(model_and_data_pdp):
    _require_spark(model_and_data_pdp)
    rf = model_and_data_pdp['rf_iris_sp']
    assembler = model_and_data_pdp['assembler_iris']
    features = model_and_data_pdp['iris_sp_df']

    def my_pred_fn(data):
        temp_df = assembler.transform(data)
        return rf.transform(temp_df)

    features_to_ignore = ['petal_length','petal_width']
    features_to_use = [f for f in list(features.columns) if f not in features_to_ignore]

    explainer = PartialDependenceExplainer(my_pred_fn, output_col='probability')
    explanation = explainer.explain(features, ignore_feats=features_to_ignore)
    return_dict = explanation.data

    # Explanation should contain all the non-ignored features
    assert list(return_dict.keys()) == features_to_use
    # Predictions should contain R^3 vectors
    assert return_dict[list(return_dict.keys())[0]]['preds'].shape[1] == 3
    # Quantiles should be R^3
    assert return_dict[list(return_dict.keys())[0]]['lower_quantile'].shape[1] == 3
    assert return_dict[list(return_dict.keys())[0]]['upper_quantile'].shape[1] == 3

    # # Check plotting doesn't crash
    # explanation.plot(filter_classes=[True, False, True], quantiles=[True, False, True])


def test_spark_regression(model_and_data_pdp):
    _require_spark(model_and_data_pdp)
    rf = model_and_data_pdp['rf_houses_sp']
    assembler = model_and_data_pdp['assembler_houses']
    features = model_and_data_pdp['houses_sp_df']

    def my_pred_fn(data):
        temp_df = assembler.transform(data)
        return rf.transform(temp_df)

    features_to_ignore = ['AveBedrms', 'Population']
    features_to_use = [f for f in list(features.columns) if f not in
            features_to_ignore]

    explainer = PartialDependenceExplainer(my_pred_fn, output_col='prediction', verbose=True, quantiles=False)
    explanation = explainer.explain(features, ignore_feats=features_to_ignore)
    return_dict = explanation.data

    # Explanation should contain all the non-ignored features
    assert list(return_dict.keys()) == features_to_use
    # Predictions should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['preds'].shape) == 1


def test_spark_regression_with_categorical(model_and_data_pdp):
    _require_spark(model_and_data_pdp)
    rf = model_and_data_pdp['rf_boston_sp']
    assembler = model_and_data_pdp['assembler_boston']
    features = model_and_data_pdp['boston_sp_df']

    # Emulate a categorical variable with strings
    features = features.withColumn('AGESTR',
                       when(col('AGE') < 20, "YOUNG")
                        .when(col('AGE') < 60, "MID")
                        .when(col('AGE') < 100, "OLD")
                        .otherwise("ELDER")
    ).drop('AGE')\
    .withColumnRenamed('AGESTR', 'AGE')

    # Make a pipeline instead of a model
    indexer = StringIndexer(inputCol="AGE", outputCol="AGEInt")
    assembler = VectorAssembler(
            inputCols=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGEInt', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
            outputCol='features'
        )
    pipe = Pipeline(stages=[indexer, assembler, rf])
    m = pipe.fit(features)

    def my_pred_fn(data):
        return m.transform(data)

    features_to_ignore = ['PTRATIO', 'B', 'LSTAT', 'CHAS', 'CRIM', 'TAX']
    features_to_use = [f for f in list(features.columns) if f not in
            features_to_ignore]

    explainer = PartialDependenceExplainer(my_pred_fn, output_col='prediction', resolution=10)
    explanation = explainer.explain(features, ignore_feats=features_to_ignore)
    return_dict = explanation.data

    # Explanation should contain all the non-ignored features
    assert list(return_dict.keys()) == features_to_use
    # Predictions should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['preds'].shape) == 1
    # Assert integrity of categorical string variables
    assert type(explanation.data['AGE']['values'][0]) == str

    # # Check plotting doesn't crash
    # explanation.plot(quantiles=False)


def test_explanation_plot(model_and_data_pdp):
    rf = model_and_data_pdp['rf_boston_sk']
    features = model_and_data_pdp['boston_pd_df']
    explainer = PartialDependenceExplainer(rf.predict)
    explanation = explainer.explain(features, ignore_feats=['PTRATIO', 'B', 'LSTAT', 'AGE'])

    assert len(explanation['CHAS'][0]) == 2 and len(explanation['CHAS'][1]) == 2

    _, ax = plt.subplots()
    explanation.plot_single('CRIM', ax=ax)
    assert ax.get_title() == 'CRIM'

    # # Check that plotting doesn't crash
    # explanation.plot(quantiles=True)


if __name__ == "__main__":
	pytest.main([__file__])
