import logging

from matplotlib import pyplot as plt
from mercury.explainability import ALEExplainer
from mercury.explainability.explainers.ale import plot_ale

import pandas as pd
import numpy as np
import pickle

import pytest


@pytest.fixture(scope='session')
def model_and_data():
    logRegModel = pickle.load(open('./tests/explainability/model_and_data/FICO_lr_model.pkl', 'rb'))
    fit_data = pd.read_csv('./tests/explainability/model_and_data_pyspark/data_ale.csv', index_col=0)
    return {
        'logRegModel': logRegModel,
        'data': fit_data,
    }

def test_ale_bad_init(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    with pytest.raises(AttributeError) as excinfo:
        ale_instance = ALEExplainer(
            lambda x: model.predict_proba(x),
            target_names=1
        )

    with pytest.raises(AttributeError) as excinfo:
        ale_instance = ALEExplainer(
            lambda x: model.predict_proba(x),
            target_names=1
        )

def test_ale_bad_explain(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']

    ale_instance = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names=['label']
    )

    with pytest.raises(ValueError) as excinfo:
        explanation = ale_instance.explain(np.array([1,2,3]))


def test_ale_explain(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    ale_instance = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names="label"
    )

    explanation = ale_instance.explain(data_pd[features])

    assert (
        explanation.ale_values[0][0][0] == pytest.approx(-0.20772727415447495,  rel=0.1, abs=0.5) and
        explanation.ale_values[0][0][1] == pytest.approx(0.20772727415447498, rel=0.1, abs=0.5) and
        explanation.ale_values[1][0][0] == pytest.approx(0.0027093534329286116, rel=0.1, abs=0.5) and
        explanation.ale_values[1][0][1] == pytest.approx(-0.0027093534329286047, rel=0.1, abs=0.5) and
        explanation.ale_values[2][0][0] == pytest.approx(0.006239338356358494, rel=0.1, abs=0.5)
    )

def test_ale_explain_ignoring(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    ale_instance = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names="label"
    )

    to_ignore = features[3:6]
    explanation = ale_instance.explain(data_pd[features], ignore_features=to_ignore)

    for e in explanation.feature_names:
        assert e not in to_ignore


def test_plot_explanation(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    ale_instance = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names='label'
    )

    explanation = ale_instance.explain(data_pd[features])

    axes = plot_ale(explanation,
                    n_cols=1,
                    fig_kw={'figwidth': 13, 'figheight': 20},
                    line_kw={'markersize': 3, 'marker': 'o', 'label': None} ,
                    sharey=None)

    assert axes.shape == (20, 1)

    fig, ax = plt.subplots(len(features))
    axes = plot_ale(explanation, features=features, targets=['label'], ax=ax)

    # Test only plot of certain features
    axes = plot_ale(explanation, features=features[3:6])
    assert axes.shape == (1,3)
