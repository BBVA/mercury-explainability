import numpy as np
import pandas as pd
import pytest
import pyspark
import pyspark.ml.regression as pysparkreg
import pyspark.ml.classification as pysparkclas

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import when, col
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from mercury.explainability.explainers.shuffle_importance import ShuffleImportanceExplainer


@pytest.fixture(scope="session")
def model_and_data():
    spark_sess = SparkSession.builder.config("k1", "v1").getOrCreate()

    houses = fetch_california_housing()
    houses_pd_df = pd.DataFrame(houses['data'], columns=houses['feature_names'])
    houses_pd_df['target'] = houses['target']
    houses_sp_df = spark_sess.createDataFrame(houses_pd_df)

    # Fit sklearn RFs
    rf_houses_sk = RandomForestRegressor().fit(houses_pd_df[['MedInc', 'HouseAge',
        'AveRooms', 'AveBedrms', 'Population', 'AveOccup','Latitude', 'Longitude']],
        houses_pd_df['target'])

    # Fit Spark RFs
    assembler_houses = VectorAssembler(
            inputCols=['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'],
            outputCol='features')
    houses_sp_df_temp = assembler_houses.transform(houses_sp_df)
    rf_houses_sp = pysparkreg.RandomForestRegressor(
        featuresCol="features",
        labelCol="target"
    ).fit(houses_sp_df_temp)

    return {
        'spark_sess': spark_sess,
        'houses_pd_df': houses_pd_df,
        'houses_sp_df': houses_sp_df.drop("features"),
        'rf_houses_sk': rf_houses_sk,
        'rf_houses_sp': rf_houses_sp,
        'assembler_houses': assembler_houses
    }


pytestmark = pytest.mark.usefixtures("model_and_data")


def test_importance_pyspark_target_doesnt_exist(model_and_data):
    rf = model_and_data['rf_houses_sp']
    assembler = model_and_data['assembler_houses']
    features = model_and_data['houses_sp_df']
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol='target')
    feat_names = [f for f in features.columns if f != 'target']

    def eval_fn(features, target):
        t = assembler.transform(features)
        t = rf.transform(t)
        return evaluator.evaluate(t)

    expl = ShuffleImportanceExplainer(eval_fn)

    with pytest.raises(ValueError):
        # Target does not exist
        explanation = expl.explain(features, 'target_not_existing')


def test_importance_pyspark_target_exists(model_and_data):
    rf = model_and_data['rf_houses_sp']
    assembler = model_and_data['assembler_houses']
    features = model_and_data['houses_sp_df']
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol='target')
    feat_names = [f for f in features.columns if f != 'target']

    def eval_fn(features, target):
        t = assembler.transform(features)
        t = rf.transform(t)
        return evaluator.evaluate(t)

    expl = ShuffleImportanceExplainer(eval_fn)
    explanation = expl.explain(features, 'target')

    assert set(explanation.data.keys()) == set(feat_names)
    assert explanation.get_importances()[0][0] == 'MedInc'

    # Check explanation doesnt crash
    explanation['MedInc']
    # Check yaxis plot contains correct label names
    ax = explanation.plot()
    assert len(ax.get_yaxis().get_ticklabels()) == len(feat_names)

def test_importance_pyspark_with_categoricals(model_and_data):
    assembler = model_and_data['assembler_houses']
    features = model_and_data['houses_sp_df']
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol='target')
    # Emulate a categorical variable
    feats = features.withColumn('AveRoomsCat',
                       when(col('AveRooms') < 2, 0)
                        .when(col('AveRooms') < 5, 1)
                        .when(col('AveRooms') < 20, 3)
                        .otherwise(10)
                ).drop('AveRooms')\
                .withColumnRenamed('AveRoomsCat', 'AveRooms')

    houses_sp_df_temp = assembler.transform(feats)

    rf= pysparkreg.RandomForestRegressor(
        featuresCol="features",
        labelCol="target"
    ).fit(houses_sp_df_temp)

    feat_names = [f for f in feats.columns if f != 'target']

    def eval_fn(features, target):
        t = assembler.transform(features)
        t = rf.transform(t)
        return evaluator.evaluate(t)

    expl = ShuffleImportanceExplainer(eval_fn)
    explanation = expl.explain(feats, 'target')

    assert set(explanation.data.keys()) == set(feat_names)
    assert explanation.get_importances()[0][0] == 'MedInc'


def test_importance_pyspark_bad_input(model_and_data):
    rf = model_and_data['rf_houses_sp']
    assembler = model_and_data['assembler_houses']
    features = model_and_data['houses_sp_df']
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol='target')
    feat_names = [f for f in features.columns if f != 'target']

    def eval_fn(features, target):
        t = assembler.transform(features)
        t = rf.transform(t)
        return evaluator.evaluate(t)

    expl = ShuffleImportanceExplainer(eval_fn)

    with pytest.raises(ValueError):
        # Explain target parameter for pysparkshould be a string
        explanation = expl.explain(features, [3,4,5,6,3])

def test_importance_pandas(model_and_data):
    rf = model_and_data['rf_houses_sk']
    houses_df = model_and_data['houses_pd_df']

    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup','Latitude', 'Longitude']
    tgt = 'target'

    feats = houses_df.loc[:, feature_names]
    targ = houses_df.loc[:, tgt]

    def eval_fn(features, target):
        pred = rf.predict(features)
        return mean_squared_error(target, pred)

    expl = ShuffleImportanceExplainer(eval_fn)
    explanation = expl.explain(feats, targ)
    assert explanation.get_importances()[0][0] == 'MedInc'

    # Same test without normalizatoin
    expl = ShuffleImportanceExplainer(eval_fn, normalize=False)
    explanation = expl.explain(feats, targ)
    assert explanation.get_importances()[0][0] == 'MedInc'


def test_importance_bad_input_pandas(model_and_data):
    rf = model_and_data['rf_houses_sk']
    houses_df = model_and_data['houses_pd_df']

    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup','Latitude', 'Longitude']
    tgt = 'target'

    feats = houses_df.loc[:, feature_names]
    targ = houses_df.loc[:, tgt]

    def eval_fn(features, target):
        pred = rf.predict(features)
        return mean_squared_error(target, pred)

    expl = ShuffleImportanceExplainer(eval_fn)

    with pytest.raises(ValueError):
        # Explain target parameter for plain python should not be a string
        explanation = expl.explain(feats, "target")

def test_importance_pandas_bug_target_col_not_filtered(model_and_data):
    rf = model_and_data['rf_houses_sk']
    houses_df = model_and_data['houses_pd_df']

    def eval_fn(data, target_col):
        X = data.loc[:, data.columns != target_col]
        Y = data.loc[:, target_col]
        Yh = rf.predict(X)
        return mean_squared_error(Y, Yh)

    expl = ShuffleImportanceExplainer(eval_fn)
    explanation = expl.explain(houses_df, 'target')
    features, importances = zip(*explanation.get_importances())

    assert 'target' not in features

