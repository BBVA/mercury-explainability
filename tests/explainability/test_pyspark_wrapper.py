from mercury.explainability.pyspark_utils import SparkWrapper

import pytest
import pandas as pd


@pytest.fixture(scope="module")
def spark_session():
    try:
        from pyspark.sql import SparkSession
        return SparkSession.builder.master("local[2]").appName("pyspark-wrapper-tests").getOrCreate()
    except Exception:
        pytest.skip("PySpark wrapper tests require a working Spark/Java runtime")


@pytest.fixture(scope="module")
def model_and_data(spark_session):
    try:
        from pyspark.ml.classification import GBTClassificationModel
        from pyspark.ml.feature import VectorAssembler

        gbt_model = GBTClassificationModel.load(
            'tests/explainability/model_and_data_pyspark/gbtModelPySpark'
        )
        assembler = VectorAssembler.load(
            'tests/explainability/model_and_data_pyspark/assemblerScaled'
        )
        data_pd = pd.read_csv(
            'tests/explainability/model_and_data_pyspark/data_pandas_red.csv',
            index_col=0,
        )
        return {
            'gbtModel': gbt_model,
            'assembler': assembler,
            'data_pd': data_pd,
        }
    except Exception as exc:
        pytest.skip(f"PySpark test assets unavailable: {exc}")


def test_predict_np(spark_session, model_and_data):
    gbtModel = model_and_data['gbtModel']
    assembler = model_and_data['assembler']
    data_pd = model_and_data['data_pd']

    shape = (1, data_pd.shape[1])
    feat_names = list(data_pd.columns)

    wrap = SparkWrapper(gbtModel,
                        feat_names,
                        spark_session,
                        model_inp_name='scaledFeatures',
                        model_out_name='probability',
                        vector_assembler=assembler
    )

    out = wrap(data_pd.head(1).values).flatten()

    assert (
        out[0] == pytest.approx(0.87631167, 0.01) and
        out[1] == pytest.approx(0.12368833, 0.01)
    )


def test_predict_pandas(spark_session, model_and_data):
    gbtModel = model_and_data['gbtModel']
    assembler = model_and_data['assembler']
    data_pd = model_and_data['data_pd']

    shape = (1, data_pd.shape[1])
    feat_names = list(data_pd.columns)

    wrap = SparkWrapper(gbtModel,
                        feat_names,
                        spark_session,
                        model_inp_name='scaledFeatures',
                        model_out_name='probability',
                        #vector_assembler=assembler
    )

    out = wrap(data_pd.head(1)).flatten()

    assert (
        out[0] == pytest.approx(0.87631167, 0.01) and
        out[1] == pytest.approx(0.12368833, 0.01)
    )


def test_normalize_outs(spark_session, model_and_data):
    gbtModel = model_and_data['gbtModel']
    assembler = model_and_data['assembler']
    data_pd = model_and_data['data_pd']

    shape = (1, data_pd.shape[1])
    feat_names = list(data_pd.columns)

    wrap = SparkWrapper(gbtModel,
                        feat_names,
                        spark_session,
                        model_inp_name='scaledFeatures',
                        model_out_name='probability',
                        probability_threshold=0.5
    )

    out = wrap(data_pd.head(1)).flatten()

    assert (
        out[0] == pytest.approx(0.87631167, 0.01) and
        out[1] == pytest.approx(0.12368833, 0.01)
    )


if __name__ == "__main__":
    pytest.main([__file__])
