import logging
from types import SimpleNamespace

from matplotlib import pyplot as plt
from mercury.explainability import ALEExplainer
from mercury.explainability.explainers.ale import plot_ale, _plot_one_ale_num
from mercury.explainability.explainers.explainer import MercuryExplainer

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


def test_ale_save_delegates_to_mercury_save(model_and_data, monkeypatch, tmp_path):
    model = model_and_data['logRegModel']

    ale_instance = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names='label'
    )

    captured = {}

    def _fake_save(self, filename):
        captured['self'] = self
        captured['filename'] = filename

    monkeypatch.setattr(MercuryExplainer, 'save', _fake_save)
    output_file = tmp_path / 'ale_explainer.pkl'
    ale_instance.save(output_file)

    assert captured['self'] is ale_instance
    assert captured['filename'] == output_file


def test_plot_ale_raises_on_unknown_feature_name(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    explanation = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names='label'
    ).explain(data_pd[features])

    with pytest.raises(ValueError, match='Feature name .* does not exist'):
        plot_ale(explanation, features=['feature_that_does_not_exist'])


def test_plot_ale_raises_on_unknown_target_name(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    explanation = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names='label'
    ).explain(data_pd[features])

    with pytest.raises(ValueError, match='Target name .* does not exist'):
        plot_ale(explanation, targets=['target_that_does_not_exist'])


def test_plot_ale_row_sharey_and_explicit_label(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    explanation = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names='label'
    ).explain(data_pd[features])

    axes = plot_ale(
        explanation,
        features=[0, 1, 2],
        sharey='row',
        n_cols=2,
        line_kw={'label': 'ALE line'}
    )

    assert axes.shape == (2, 2)


def test_plot_ale_single_feature_with_axis_object(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    explanation = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names='label'
    ).explain(data_pd[features])

    fig, ax = plt.subplots()
    axes = plot_ale(explanation, features=[0], ax=ax)
    assert axes.size == 1


def test_plot_ale_raises_when_axes_array_too_small(model_and_data):
    model = model_and_data['logRegModel']
    data_pd = model_and_data['data']
    features = [c for c in list(data_pd.columns) if c not in ['label']]

    explanation = ALEExplainer(
        lambda x: model.predict_proba(x),
        target_names='label'
    ).explain(data_pd[features])

    fig, ax = plt.subplots(1)
    with pytest.raises(ValueError, match='Expected ax to have'):
        plot_ale(explanation, features=[0, 1], ax=np.array([ax]))


def test_plot_one_ale_num_handles_shorter_feature_values():
    exp = SimpleNamespace(
        feature_values=[np.array([0.0, 1.0])],
        ale_values=[np.array([[0.1], [0.2], [0.3]])],
        constant_value=0.0,
        feature_deciles=[np.array([0.0, 0.5, 1.0])],
        feature_names=['f0'],
        target_names=np.array(['label'])
    )

    ax = _plot_one_ale_num(
        exp=exp,
        feature=0,
        targets=[0],
        constant=False,
        ax=None,
        legend=True,
        line_kw={'label': None, 'marker': 'o', 'markersize': 3}
    )

    assert ax.get_xlabel() == 'f0'


def test_plot_one_ale_num_handles_shorter_ale_values():
    fig, ax = plt.subplots()
    exp = SimpleNamespace(
        feature_values=[np.array([0.0, 1.0, 2.0])],
        ale_values=[np.array([[0.1], [0.2]])],
        constant_value=0.0,
        feature_deciles=[np.array([0.0, 1.0, 2.0])],
        feature_names=['f0'],
        target_names=np.array(['label'])
    )

    out_ax = _plot_one_ale_num(
        exp=exp,
        feature=0,
        targets=[0],
        constant=False,
        ax=ax,
        legend=True,
        line_kw={'label': None, 'marker': 'o', 'markersize': 3}
    )

    assert out_ax is ax


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	pytest.main([__file__])
