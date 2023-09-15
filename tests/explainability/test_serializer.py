from mercury.explainability.explainers import (
    AnchorsWithImportanceExplainer,
    ALEExplainer,
    MercuryExplainer
)
import pytest
import pickle
import pandas as pd
import numpy as np
import os


@pytest.fixture(scope="session")
def model_and_data_anchors():
    logRegModel = pickle.load(open('tests/explainability/model_and_data/FICO_lr_model.pkl', 'rb'))
    fit_data = pd.read_csv('tests/explainability/model_and_data/fit_data_red.csv', index_col=0)
    explain_data = pd.read_csv('tests/explainability/model_and_data/explain_data.csv', index_col=0)
    return {
        'logRegModel': logRegModel,
        'fit_data': fit_data,
        'explain_data': explain_data
    }

@pytest.fixture(scope='session')
def model_and_data_ale():
    logRegModel = pickle.load(open('./tests/explainability/model_and_data/FICO_lr_model.pkl', 'rb'))
    fit_data = pd.read_csv('./tests/explainability/model_and_data_pyspark/data_ale.csv', index_col=0)
    return {
        'logRegModel': logRegModel,
        'data': fit_data,
    }

pytestmark = pytest.mark.usefixtures("model_and_data")

def test_serializer_explainer(model_and_data_anchors):
    """
        Testing out that the explainers are properly saved and then loaded
        back.
    """
    logRegModel = model_and_data_anchors['logRegModel']
    fit_data = model_and_data_anchors['fit_data']
    explain_data = model_and_data_anchors['explain_data']
    feature_names = list(explain_data.columns)
    
    TEST_FILE = "/tmp/explainer.pkl"

    anchorsExtendedExplainer = AnchorsWithImportanceExplainer(
        train_data=fit_data,
        predict_fn=logRegModel.predict_proba,
        feature_names=feature_names
    )
    anchorsExtendedExplainer.save(TEST_FILE)
    assert os.path.isfile(TEST_FILE), "File does not exist"

    anchorsExtendedExplainer_recovered = MercuryExplainer.load(TEST_FILE)
    assert type(anchorsExtendedExplainer_recovered) ==\
         AnchorsWithImportanceExplainer, "Bad load"
    
    os.remove(TEST_FILE)

def test_serializer_anchors_with_importance_explainer(model_and_data_anchors):

    logRegModel = model_and_data_anchors['logRegModel']
    fit_data = model_and_data_anchors['fit_data']
    explain_data = model_and_data_anchors['explain_data']
    feature_names = list(explain_data.columns)

    TEST_FILE = "/tmp/explainer_anchor.pkl"

    explainer = AnchorsWithImportanceExplainer(
        train_data=fit_data,
        predict_fn=logRegModel.predict_proba,
        feature_names=feature_names
    )
    explainer.save(TEST_FILE)

    explainer_loaded = MercuryExplainer.load(TEST_FILE)
    assert isinstance(explainer_loaded, AnchorsWithImportanceExplainer)
    assert explainer.params == explainer_loaded.params
    assert explainer.feature_values == explainer_loaded.feature_values

    # We are able to execute explain
    explanation_loaded = explainer_loaded.explain(fit_data.values[0])

    os.remove(TEST_FILE)

def test_serializer_ale_explainer(model_and_data_ale):

    model = model_and_data_ale['logRegModel']
    data_pd = model_and_data_ale['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    TEST_FILE = "/tmp/explainer_ale.pkl"

    explainer = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names="label"
    )
    explainer.save(TEST_FILE)

    explainer_loaded = MercuryExplainer.load(TEST_FILE)
    isinstance(explainer_loaded, ALEExplainer)

    # Check explanations
    explanation = explainer.explain(data_pd[features])
    explanation_loaded = explainer_loaded.explain(data_pd[features])

    for i in range(len(explanation.data)):
        assert np.all(explanation.data['ale_values'][i] == explanation_loaded.data['ale_values'][i])

    os.remove(TEST_FILE)