import numpy as np
import pandas as pd
from mercury.explainability.explainers import CounterfactualExplainer, CounterfactualProtoExplainer
import pytest
import pickle
import tensorflow as tf
# from tensorflow.python.framework import ops

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


# def test_cf_proto_explain(model_and_data):
#     # Test it does not crash during __init__ (where fit is done) and
#     # that returns some counterfactual.
#     logRegModel = model_and_data['logRegModel']
#     fit_data = model_and_data['fit_data']
#     explain_data = model_and_data['explain_data']
#     feature_names = list(explain_data.columns)

#     cfExplainer = CounterfactualProtoExplainer(
#         predict_fn=logRegModel.predict_proba,
#         train_data=fit_data,
#         use_kdtree=True,
#     )

#     explanation = cfExplainer.explain(explain_data.head(1).values)

#     cfs = False
#     for c in list(explanation['data']['all'].values()):
#         if len(c) > 0:
#             cfs = True
#             break

#     assert cfs


# def test_cfproto_feature_importance(model_and_data):
#     logRegModel = model_and_data['logRegModel']
#     fit_data = model_and_data['fit_data']
#     explain_data = model_and_data['explain_data']
#     feature_names = list(explain_data.columns)

#     predict_fn = lambda x: logRegModel.predict_proba(x)

#     shape = (1,) + fit_data.shape[1:]

#     feature_range = (
#         fit_data.min(axis=0),
#         fit_data.max(axis=0)
#     )

#     c_init = 1.
#     c_steps = 10
#     eps = (1e-2, 1e-2)
#     theta = 10
#     max_iter = 1000

#     cfprotoExtendedExplainer = CounterfactualProtoExplainer(
#         predict_fn=predict_fn,
#         train_data=fit_data,
#         shape=shape,
#         feature_names=feature_names,
#         feature_range=feature_range,
#         c_init=c_init,
#         c_steps=c_steps,
#         max_iterations=max_iter,
#         theta=theta,
#         eps=eps,
#         update_num_grad=1
#     )

#     explanations = cfprotoExtendedExplainer.get_feature_importance(
#         explain_data.head(3)
#     )

#     assert (
#         explanations.importances[0][0] == 'ExternalRiskEstimate' and
#         explanations.importances[0][1] == 1.0 and
#         explanations.importances[0][2] == -1.0 and
#         explanations.importances[1][0] == 'NetFractionRevolvingBurden' and
#         explanations.importances[1][1] >= 0.5 and
#         explanations.importances[1][2] == 1.0
#     )


def test_cf_proto_explain(model_and_data):
    # Skipped due to alibi (which is optional) expecting tensorflow 1.xx
    pass


def test_cfproto_feature_importance(model_and_data):
    # Skipped due to alibi (which is optional) expecting tensorflow 1.xx
    pass
