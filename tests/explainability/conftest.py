import logging

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as Sklearn_KMeans
from sklearn.preprocessing import StandardScaler
from pyspark.ml.classification import GBTClassificationModel
from sklearn.pipeline import Pipeline as SklearnPipeline
import pyspark.ml.regression as pysparkreg
import pyspark.ml.classification as pysparkclas
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml import Pipeline


import os
import pandas as pd
import pytest

@pytest.fixture(scope='package')
def spark_context():
    conf = (SparkConf().setMaster('local[2]').setAppName('Alibi Tests'))
    sc = SparkContext(conf=conf)

    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)

    return sc

@pytest.fixture(scope='package')
def spark_session(spark_context):
    return SparkSession.builder.appName('Alibi Tests').getOrCreate()

@pytest.fixture(scope="module")
def model_and_data(spark_session):
    gbtModel = GBTClassificationModel.load('./tests/explainability/model_and_data_pyspark/gbtModelPySpark')
    assembler = VectorAssembler.load('./tests/explainability/model_and_data_pyspark/assemblerScaled')
    data_pd = pd.read_csv('./tests/explainability/model_and_data_pyspark/data_pandas_red.csv', index_col=0)
    data_pyspark = spark_session.createDataFrame(data_pd)
    data_preproc = assembler.transform(data_pyspark)
    return {
        'gbtModel': gbtModel,
        'assembler': assembler,
        'data_pd': data_pd,
        'data_pyspark': data_pyspark,
        'data_preproc': data_preproc
    }

@pytest.fixture(scope='module')
def model_and_data_ale(spark_session):
    gbtModel = GBTClassificationModel.load('./tests/explainability/model_and_data_pyspark/gbtModelPySpark')
    assembler = VectorAssembler.load('./tests/explainability/model_and_data_pyspark/assemblerScaled')
    data_pd = pd.read_csv('./tests/explainability/model_and_data_pyspark/data_ale.csv', index_col=0)
    data_pyspark = spark_session.createDataFrame(data_pd)
    data_preproc = assembler.transform(data_pyspark)
    return {
        'gbtModel': gbtModel,
        'assembler': assembler,
        'data_pd': data_pd,
        'data_pyspark': data_pyspark,
        'data_preproc': data_preproc
    }

@pytest.fixture(scope="session", autouse=True)
def env_var_patching():
    os.environ["MOMA_ENV"] = "test"
    os.environ["MERCURY_LOGGING_DISABLE"] = "1"


@pytest.fixture(scope="module")
def model_and_data_pdp(spark_session):
    spark_sess = spark_session

    iris = load_iris()
    houses = fetch_california_housing()
    boston = boston = load_boston()

    houses_pd_df = pd.DataFrame(houses['data'], columns=houses['feature_names'])
    houses_pd_df['target'] = houses['target']
    iris_pd_df = pd.DataFrame(iris['data'], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    iris_pd_df['target'] = iris['target']
    boston_pd_df = pd.read_csv("model_and_data/boston.csv")

    houses_sp_df = spark_sess.createDataFrame(houses_pd_df)
    iris_sp_df = spark_sess.createDataFrame(iris_pd_df)
    boston_sp_df = spark_sess.createDataFrame(boston_pd_df)

    # Fit sklearn RFs
    rf_iris_sk = RandomForestClassifier().fit(iris_pd_df[['sepal_length', 'sepal_width',
        'petal_length', 'petal_width']], iris_pd_df['target'])
    rf_houses_sk = RandomForestRegressor().fit(houses_pd_df[['MedInc', 'HouseAge',
        'AveRooms', 'AveBedrms', 'Population', 'AveOccup','Latitude', 'Longitude']],
        houses_pd_df['target'])
    rf_boston_sk = RandomForestRegressor().fit(
            boston_pd_df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                          'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']],
            boston_pd_df['target']
    )

    # Fit Spark RFs
    assembler_iris = VectorAssembler(inputCols=['sepal_length','sepal_width','petal_length','petal_width'],
                            outputCol='features')
    iris_sp_df_temp = assembler_iris.transform(iris_sp_df)
    rf_iris_sp = pysparkclas.RandomForestClassifier(featuresCol="features",
            labelCol="target").fit(iris_sp_df_temp)

    assembler_houses = VectorAssembler(
            inputCols=['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'],
            outputCol='features')
    houses_sp_df_temp = assembler_houses.transform(houses_sp_df)
    rf_houses_sp = pysparkreg.RandomForestRegressor(
        featuresCol="features",
        labelCol="target"
    ).fit(houses_sp_df_temp)

    assembler_boston = VectorAssembler(
        inputCols=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
        outputCol='features'
    )
    boston_sp_df_temp = assembler_boston.transform(boston_sp_df)
    rf_boston_sp = pysparkreg.RandomForestRegressor(
        featuresCol='features',
        labelCol='target'
    ).fit(boston_sp_df_temp)

    return {
        'spark_sess': spark_sess,
        'iris_pd_df': iris_pd_df.loc[:, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        'houses_pd_df': houses_pd_df.loc[:, ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude', 'Longitude']],
        'boston_pd_df': boston_pd_df.loc[:, ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']],
        'iris_sp_df': iris_sp_df.drop("features").drop('target'),
        'houses_sp_df': houses_sp_df.drop("features").drop('target'),
        'boston_sp_df': boston_sp_df.drop("target"),
        'rf_iris_sk': rf_iris_sk,
        'rf_houses_sk': rf_houses_sk,
        'rf_boston_sk': rf_boston_sk,
        'rf_iris_sp': rf_iris_sp,
        'assembler_iris':assembler_iris,
        'rf_houses_sp': rf_houses_sp,
        'assembler_houses': assembler_houses,
        'assembler_boston': assembler_boston,
        'rf_boston_sp': rf_boston_sp
    }

@pytest.fixture(scope="module")
def model_and_data_cte(spark_session):

    # Generate Dataset
    K = 3
    random_state = 42
    x_data, _ = make_blobs(n_samples=1000, n_features=2, centers=K, cluster_std=2.5, random_state=random_state)

    # K-means with pandas and sklearn
    features_names = ["feature_1", "feature_2"]
    pandas_df = pd.DataFrame(x_data, columns=features_names)
    sk_kmeans = Sklearn_KMeans(K, random_state=42)
    sk_kmeans.fit(x_data)

    # K-means sklearn pipeline
    sk_pipeline = SklearnPipeline(steps=[("scaler", StandardScaler()), ("kmeans", Sklearn_KMeans(K, random_state=random_state))])
    sk_pipeline.fit(x_data)

    # K-means with spark dataframes (spark pipeline)
    spark_df = spark_session.createDataFrame(pandas_df)
    assembler = VectorAssembler(inputCols=features_names, outputCol="features")
    spark_kmeans = SparkKMeans(k=K, seed=random_state)
    spark_pipeline = Pipeline(stages=[assembler, spark_kmeans])
    spark_pipeline_model = spark_pipeline.fit(spark_df)

    # K-means with spark dataframes (no pipeline)
    assembler = VectorAssembler(inputCols=features_names, outputCol="features")
    spark_df_2 = assembler.transform(spark_df).select("features")
    spark_kmeans = SparkKMeans(k=K, seed=random_state)
    spark_kmeans_model = spark_kmeans.fit(spark_df_2)

    return {
        'spark_sess': spark_session,
        'pandas_df': pandas_df,
        'sk_kmeans': sk_kmeans,
        'sk_pipeline': sk_pipeline,
        'spark_df': spark_df,
        'spark_pipeline_model': spark_pipeline_model,
        'spark_df_2': spark_df_2,
        'spark_kmeans_model': spark_kmeans_model,
        'K': K
    }