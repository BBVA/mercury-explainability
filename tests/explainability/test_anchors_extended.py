import numpy as np
import pandas as pd
from mercury.explainability.explainers import AnchorsWithImportanceExplainer
import pytest
import pickle
import tensorflow as tf

@pytest.fixture(scope="session")
def model_and_data():
    logRegModel = pickle.load(open('./tests/explainability/model_and_data/FICO_lr_model.pkl', 'rb'))
    fit_data = pd.read_csv('./tests/explainability/model_and_data/fit_data_red.csv', index_col=0)
    explain_data = pd.read_csv('./tests/explainability/model_and_data/explain_data.csv', index_col=0)
    return {
        'logRegModel': logRegModel,
        'fit_data': fit_data,
        'explain_data': explain_data
    }


pytestmark = pytest.mark.usefixtures("model_and_data")


def test_anchors_build(model_and_data):
    """
        Testing out that we can obtain explanations and they make sense.
        For that we have to fit the anchors model and provide a seed to
        avoid randomness.
    """
    logRegModel = model_and_data['logRegModel']
    fit_data = model_and_data['fit_data']
    explain_data = model_and_data['explain_data']
    
    anchorsExtendedExplainer = AnchorsWithImportanceExplainer(
        predict_fn=logRegModel.predict_proba,
        train_data=fit_data,
        disc_perc=[10,20,30,40,50,60,70,80,90]
    )

    with pytest.raises(AttributeError) as excinfo:
        anchorsExtendedExplainer = AnchorsWithImportanceExplainer(
            predict_fn=logRegModel.predict_proba,
            train_data='wrong data',
            disc_perc=[10,20,30,40,50,60,70,80,90]
        )
    
    with pytest.raises(AttributeError) as excinfo:
        anchorsExtendedExplainer = AnchorsWithImportanceExplainer(
            predict_fn=logRegModel.predict_proba,
            train_data=fit_data,
            categorical_names=[]
        )

def test_anchors_fit_and_explain_precision(model_and_data):
    """
        Testing out that we can obtain explanations and they make sense.
        For that we have to fit the anchors model and provide a seed to
        avoid randomness.
    """
    logRegModel = model_and_data['logRegModel']
    fit_data = model_and_data['fit_data']
    explain_data = model_and_data['explain_data']
    
    anchorsExtendedExplainer = AnchorsWithImportanceExplainer(
        predict_fn=logRegModel.predict_proba,
        train_data=fit_data,
        disc_perc=[10,20,30,40,50,60,70,80,90]
    )

    np.random.seed(42)
    explanation = anchorsExtendedExplainer.explain(
        explain_data.head(1).values, threshold=0.95
    )

    assert explanation.data['precision'] > 0.95

def test_anchors_fit_and_explain_coverage(model_and_data):
    """
        Testing out that we can obtain explanations and they make sense.
        For that we have to fit the anchors model and provide a seed to
        avoid randomness.
    """
    logRegModel = model_and_data['logRegModel']
    fit_data = model_and_data['fit_data']
    explain_data = model_and_data['explain_data']
    
    anchorsExtendedExplainer = AnchorsWithImportanceExplainer(
        predict_fn=logRegModel.predict_proba,
        train_data=fit_data,
        disc_perc=[10,20,30,40,50,60,70,80,90]
    )

    np.random.seed(42)
    explanation = anchorsExtendedExplainer.explain(
        explain_data.head(1).values, threshold=0.95
    )
    assert explanation.data['coverage'] == pytest.approx(0.42, 0.01)

def test_anchors_feature_importance_obtention(model_and_data):
    """
        Testing out that we can obtain explanations and they make sense.
        For that we have to fit the anchors model and provide a seed to
        avoid randomness.
    """
    logRegModel = model_and_data['logRegModel']
    fit_data = model_and_data['fit_data']
    explain_data = model_and_data['explain_data']
    
    anchorsExtendedExplainer = AnchorsWithImportanceExplainer(
        predict_fn=logRegModel.predict_proba,
        train_data=fit_data,
        disc_perc=[10,20,30,40,50,60,70,80,90]
    )

    np.random.seed(42)
    anchorsExplanations = anchorsExtendedExplainer.get_feature_importance(
        explain_data.head(10), print_every=10, print_explanations=True
    )

    anchorsInterpretation = anchorsExplanations.interpret_explanations(n_important_features=3)
    assert (
        'ExternalRiskEstimate' in anchorsInterpretation and
        'NetFractionRevolvingBurden' in anchorsInterpretation and
        'AverageMInFile' in anchorsInterpretation
    )

def test_anchors_feature_importance_obtention_top_5(model_and_data):
    """
        Testing out that we can obtain explanations and they make sense.
        For that we have to fit the anchors model and provide a seed to
        avoid randomness.
    """
    logRegModel = model_and_data['logRegModel']
    fit_data = model_and_data['fit_data']
    explain_data = model_and_data['explain_data']
    
    anchorsExtendedExplainer = AnchorsWithImportanceExplainer(
        predict_fn=logRegModel.predict_proba,
        train_data=fit_data,
        disc_perc=[10,20,30,40,50,60,70,80,90]
    )

    np.random.seed(42)
    anchorsExplanations = anchorsExtendedExplainer.get_feature_importance(
        explain_data.head(10)
    )

    anchorsInterpretation = anchorsExplanations.interpret_explanations(n_important_features=5)

    # I don't include the last two features because they only have a frequency of 1.

    assert (
        'ExternalRiskEstimate' in anchorsInterpretation and
        'NetFractionRevolvingBurden' in anchorsInterpretation and
        'AverageMInFile' in anchorsInterpretation
    )
