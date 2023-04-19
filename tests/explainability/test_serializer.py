from mercury.explainability.explainers import (
    AnchorsWithImportanceExplainer,
    MercuryExplainer
)
import pytest
import pickle
import pandas as pd
import os


@pytest.fixture(scope="session")
def model_and_data():
    logRegModel = pickle.load(open('tests/explainability/model_and_data/FICO_lr_model.pkl', 'rb'))
    fit_data = pd.read_csv('tests/explainability/model_and_data/fit_data_red.csv', index_col=0)
    explain_data = pd.read_csv('tests/explainability/model_and_data/explain_data.csv', index_col=0)
    return {
        'logRegModel': logRegModel,
        'fit_data': fit_data,
        'explain_data': explain_data
    }


pytestmark = pytest.mark.usefixtures("model_and_data")

def test_serializer_explainer(model_and_data):
    """
        Testing out that the explainers are properly saved and then loaded
        back.
    """
    logRegModel = model_and_data['logRegModel']
    fit_data = model_and_data['fit_data']
    explain_data = model_and_data['explain_data']
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