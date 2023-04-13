from mercury.explainability.pyspark_utils import SparkWrapper

import pytest

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
