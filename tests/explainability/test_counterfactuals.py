import numpy as np
import pandas as pd
from mercury.explainability.explainers import CounterfactualExplainer
import pytest
import pickle
import tensorflow as tf


@pytest.fixture(scope='session')
def model_and_data():
    tf.compat.v1.disable_eager_execution()
    logRegModel = pickle.load(open('./tests/explainability/model_and_data/FICO_lr_model.pkl', 'rb'))
    fit_data = pd.read_csv('./tests/explainability/model_and_data/fit_data_red.csv', index_col=0)
    explain_data = pd.read_csv('./tests/explainability/model_and_data/explain_data.csv', index_col=0)
    return {
        'logRegModel': logRegModel,
        'fit_data': fit_data,
        'explain_data': explain_data
    }


pytestmark = pytest.mark.usefixtures('model_and_data')


def test_cf_explain(model_and_data):
    # Test it does not crash during __init__ (where fit is done) and 
    # that returns some counterfactual.
    model = model_and_data['logRegModel']
    fit_data = model_and_data['fit_data']
    explain_data = model_and_data['explain_data']
    feature_names = list(explain_data.columns)

    cfExplainer = CounterfactualExplainer(
        predict_fn=model.predict_proba,
        feature_names=feature_names
    )

    cf_explanation = cfExplainer.explain(explain_data.head(1).values)

    assert (
        cf_explanation.data['cf']['X'][0][0] == pytest.approx(1.3380102, 0.01) and
        cf_explanation.data['cf']['X'][0][1] == pytest.approx(-1.5059024, 0.01)
    )

def test_cf_feature_importance(model_and_data):
    # Test feat importance
    model = model_and_data['logRegModel']
    fit_data = model_and_data['fit_data']
    explain_data = model_and_data['explain_data']
    feature_names = list(explain_data.columns)

    tf.compat.v1.reset_default_graph()
    cfExplainer = CounterfactualExplainer(
        predict_fn=model.predict_proba,
        feature_names=feature_names
    )

    explanations = cfExplainer.get_feature_importance(
        explain_data.head(3)
    )

    assert (
        explanations.importances[0][0] == 'ExternalRiskEstimate' and
        explanations.count_diffs_norm['ExternalRiskEstimate'] == 1.0
    )

