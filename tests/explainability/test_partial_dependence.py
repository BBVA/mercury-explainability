
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mercury.explainability import PartialDependenceExplainer

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import when, col

import pytest

pytestmark = pytest.mark.usefixtures("model_and_data_pdp")

def test_pandas_classification(model_and_data_pdp):
    """ Tests the explainer for  multinomial classification task using "plain Python"
        models
    """
    rf = model_and_data_pdp['rf_iris_sk']
    features = model_and_data_pdp['iris_pd_df']

    explainer = PartialDependenceExplainer(rf.predict_proba)
    explanation = explainer.explain(features)
    return_dict = explanation.data

    # Explanation should contain all the features
    assert list(return_dict.keys()) == list(features.columns)
    # Predictions should contain R^3 vectors
    assert return_dict[list(return_dict.keys())[0]]['preds'].shape[1] == 3
    # Quantiles should be R^3
    assert return_dict[list(return_dict.keys())[0]]['lower_quantile'].shape[1] == 3
    assert return_dict[list(return_dict.keys())[0]]['upper_quantile'].shape[1] == 3

    # Check plotting doesnt crash
    explanation.plot()

def test_pandas_regression(model_and_data_pdp):
    rf = model_and_data_pdp['rf_houses_sk']
    features = model_and_data_pdp['houses_pd_df']
    explainer = PartialDependenceExplainer(rf.predict, verbose=True)

    features_to_ignore = ['HouseAge', 'AveBedrms', 'Population']
    features_to_use = [f for f in list(features.columns) if f not in
            features_to_ignore]

    explanation = explainer.explain(features, ignore_feats=features_to_ignore)

    return_dict = explanation.data
    # Explanation should contain all the non-ignored features
    assert list(return_dict.keys()) == features_to_use
    # Predictions should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['preds'].shape) == 1
    # Quantiles should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['lower_quantile'].shape) == 1
    assert len(return_dict[list(return_dict.keys())[0]]['upper_quantile'].shape) == 1

    # Check plotting doesnt crash
    explanation.plot()


def test_pandas_regression_with_categoricals(model_and_data_pdp):
    rf = model_and_data_pdp['rf_boston_sk']
    features = model_and_data_pdp['boston_pd_df']
    explainer = PartialDependenceExplainer(rf.predict, quantiles=False)
    explanation = explainer.explain(features, ignore_feats=['PTRATIO', 'B', 'LSTAT', 'AGE'])
    return_dict = explanation.data
    # Predictions should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['preds'].shape) == 1

    assert len(return_dict['NOX']['values']) == 50

    # Check plotting doesnt crash
    explanation.plot()


def test_spark_classification(model_and_data_pdp):
    rf = model_and_data_pdp['rf_iris_sp']
    assembler = model_and_data_pdp['assembler_iris']
    features = model_and_data_pdp['iris_sp_df']

    def my_pred_fn(data):
        temp_df = assembler.transform(data)
        return rf.transform(temp_df)

    features_to_ignore = ['petal_length','petal_width']
    features_to_use = [f for f in list(features.columns) if f not in features_to_ignore]

    explainer = PartialDependenceExplainer(my_pred_fn, output_col='probability')
    explanation = explainer.explain(features, ignore_feats=features_to_ignore)
    return_dict = explanation.data

    # Explanation should contain all the non-ignored features
    assert list(return_dict.keys()) == features_to_use
    # Predictions should contain R^3 vectors
    assert return_dict[list(return_dict.keys())[0]]['preds'].shape[1] == 3
    # Quantiles should be R^3
    assert return_dict[list(return_dict.keys())[0]]['lower_quantile'].shape[1] == 3
    assert return_dict[list(return_dict.keys())[0]]['upper_quantile'].shape[1] == 3

    # Check plotting doesnt crash
    explanation.plot(filter_classes=[True, False, True], quantiles=[True, False, True])


def test_spark_regression(model_and_data_pdp):
    rf = model_and_data_pdp['rf_houses_sp']
    assembler = model_and_data_pdp['assembler_houses']
    features = model_and_data_pdp['houses_sp_df']

    def my_pred_fn(data):
        temp_df = assembler.transform(data)
        return rf.transform(temp_df)

    features_to_ignore = ['AveBedrms', 'Population']
    features_to_use = [f for f in list(features.columns) if f not in
            features_to_ignore]

    explainer = PartialDependenceExplainer(my_pred_fn, output_col='prediction', verbose=True, quantiles=False)
    explanation = explainer.explain(features, ignore_feats=features_to_ignore)
    return_dict = explanation.data

    # Explanation should contain all the non-ignored features
    assert list(return_dict.keys()) == features_to_use
    # Predictions should contain real numbers
    assert len(return_dict[list(return_dict.keys())[0]]['preds'].shape) == 1


def test_explanation_plot(model_and_data_pdp):
    rf = model_and_data_pdp['rf_boston_sk']
    features = model_and_data_pdp['boston_pd_df']
    explainer = PartialDependenceExplainer(rf.predict)
    explanation = explainer.explain(features, ignore_feats=['PTRATIO', 'B', 'LSTAT', 'AGE'])

    assert len(explanation['CHAS'][0]) == 2 and len(explanation['CHAS'][1]) == 2

    _, ax = plt.subplots()
    explanation.plot_single('CRIM', ax=ax)
    assert ax.get_title() == 'CRIM'

    # Check that plotting doesnt crash
    explanation.plot(quantiles=True)


