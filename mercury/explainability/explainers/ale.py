import typing as TP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import copy
import math

from typing import no_type_check
from itertools import count
from matplotlib.gridspec import GridSpec
from alibi.api.interfaces import Explanation, Explainer
from alibi.api.defaults import DEFAULT_META_ALE, DEFAULT_DATA_ALE
from alibi.explainers.ale import get_quantiles, bisect_fun, minimum_satisfied, adaptive_grid, ale_num

from .explainer import MercuryExplainer


class ALEExplainer(Explainer, MercuryExplainer):
    """
    Accumulated Local Effects for tabular datasets. Current implementation supports first order
    feature effects of numerical features.

    Args:
        predictor:
            A callable that takes in an NxF array as input and outputs an NxT array (N - number of
            data points, F - number of features, T - number of outputs/targets (e.g. 1 for single output
            regression, >=2 for classification).
        target_names:
            A list of target/output names used for displaying results.
    """

    def __init__(self, predictor: TP.Callable, target_names: TP.Union[TP.List[str], str]) -> None:
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ALE))

        if (not isinstance(target_names, list) and
                not isinstance(target_names, str)):
            raise AttributeError('The attribute target_names should be a string or list.')

        if type(target_names) == str:
            target_names = [target_names]

        self.predictor = predictor
        self.target_names = target_names

    def explain(self, X: pd.DataFrame, min_bin_points: int = 4, ignore_features: list = []) -> Explanation:
        """
        Calculate the ALE curves for each feature with respect to the dataset `X`.

        Args:
            X:
                An NxF tabular dataset used to calculate the ALE curves. This is typically the training dataset
                or a representative sample.
            min_bin_points:
                Minimum number of points each discretized interval should contain to ensure more precise
                ALE estimation.
            ignore_features:
                Features that will be ignored while computing the ALE curves. Useful for reducing computing time
                if there are predictors we dont care about.

        Returns:
            An `Explanation` object containing the data and the metadata of the calculated ALE curves.

        """
        self.meta['params'].update(min_bin_points=min_bin_points)

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X must be a pandas DataFrame')

        features = list(X.columns)
        n_features =len(features)

        self.feature_names = np.array(features)
        self.target_names = np.array(self.target_names)

        feature_values = []
        ale_values = []
        ale0 = []
        feature_deciles = []

        X = X[features].values

        # TODO: use joblib to paralelise?
        for feature, feat_name in enumerate(self.feature_names):
            if feat_name not in ignore_features:
                q, ale, a0 = ale_num(
                    self.predictor,
                    X=X,
                    feature=feature,
                    min_bin_points=min_bin_points
                )
                deciles = get_quantiles(X[:, feature], num_quantiles=11)

                feature_values.append(q)
                ale_values.append(ale)
                ale0.append(a0)
                feature_deciles.append(deciles)

        constant_value = self.predictor(X).mean()
        # TODO: an ALE plot ideally requires a rugplot to gauge density of instances in the feature space.
        # I've replaced this with feature deciles which is coarser but has constant space complexity
        # as opposed to a rugplot. Alternatively, could consider subsampling to produce a rug with some
        # maximum number of points.
        return self._build_explanation(
            ale_values=ale_values,
            ale0=ale0,
            constant_value=constant_value,
            feature_values=feature_values,
            feature_deciles=feature_deciles,
            ignore_features=ignore_features
        )

    def _build_explanation(self,
                          ale_values: TP.List[np.ndarray],
                          ale0: TP.List[np.ndarray],
                          constant_value: float,
                          feature_values: TP.List[np.ndarray],
                          feature_deciles: TP.List[np.ndarray],
                          ignore_features: TP.List = []) -> Explanation:
        """
        Helper method to build the Explanation object.
        """
        # TODO decide on the format for these lists of arrays
        # Currently each list element relates to a feature and each column relates to an output dimension,
        # this is different from e.g. SHAP but arguably more convenient for ALE.

        data = copy.deepcopy(DEFAULT_DATA_ALE)
        data.update(
            ale_values=ale_values,
            ale0=ale0,
            constant_value=constant_value,
            feature_values=feature_values,
            feature_names=[x for x in self.feature_names if x not in ignore_features],
            target_names=self.target_names,
            feature_deciles=feature_deciles
        )

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

@no_type_check
def plot_ale(exp: Explanation,
             features: TP.Union[TP.List[TP.Union[int, str]], str] = 'all',
             targets: TP.Union[TP.List[TP.Union[int, str]], str] = 'all',
             n_cols: int = 3,
             sharey: str = 'all',
             constant: bool = False,
             ax: TP.Union['plt.Axes', np.ndarray, None] = None,
             line_kw: TP.Optional[dict] = None,
             fig_kw: TP.Optional[dict] = None) -> 'np.ndarray':
    """
    Plot ALE curves on matplotlib axes.

    Args:
        exp: An `Explanation` object produced by a call to the `ALE.explain` method.
        features: A list of features for which to plot the ALE curves or `all` for all features.
            Can be a mix of integers denoting feature index or strings denoting entries in
            `exp.feature_names`. Defaults to 'all'.
        targets: A list of targets for which to plot the ALE curves or `all` for all targets.
            Can be a mix of integers denoting target index or strings denoting entries in
            `exp.target_names`. Defaults to 'all'.
        n_cols: Number of columns to organize the resulting plot into.
        sharey: A parameter specifying whether the y-axis of the ALE curves should be on the same scale
            for several features. Possible values are `all`, `row`, `None`.
        constant: A parameter specifying whether the constant zeroth order effects should be added to the
            ALE first order effects.
        ax: A `matplotlib` axes object or a numpy array of `matplotlib` axes to plot on.
        line_kw: Keyword arguments passed to the `plt.plot` function.
        fig_kw: Keyword arguments passed to the `fig.set` function.

    Returns:
        An array of matplotlib axes with the resulting ALE plots.

    """
    # line_kw and fig_kw values
    default_line_kw = {'markersize': 3, 'marker': 'o', 'label': None}
    if line_kw is None:
        line_kw = {}
    line_kw = {**default_line_kw, **line_kw}

    default_fig_kw = {'tight_layout': 'tight'}
    if fig_kw is None:
        fig_kw = {}
    fig_kw = {**default_fig_kw, **fig_kw}

    if features == 'all':
        features = range(0, len(exp.feature_names))
    else:
        for ix, f in enumerate(features):
            if isinstance(f, str):
                try:
                    exp.feature_names.index(f)
                except ValueError:
                    raise ValueError("Feature name {} does not exist.".format(f))
    n_features = len(features)

    if targets == 'all':
        targets = range(0, len(exp.target_names))
    else:
        for ix, t in enumerate(targets):
            if isinstance(t, str):
                try:
                    t = np.argwhere(exp.target_names == t).item()
                except ValueError:
                    raise ValueError("Target name {} does not exist.".format(t))
            targets[ix] = t

    # make axes
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(ax, plt.Axes) and n_features != 1:
        ax.set_axis_off()  # treat passed axis as a canvas for subplots
        fig = ax.figure
        n_cols = min(n_cols, n_features)
        n_rows = math.ceil(n_features / n_cols)

        axes = np.empty((n_rows, n_cols), dtype=np.object_)
        axes_ravel = axes.ravel()
        # gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=ax.get_subplotspec())
        gs = GridSpec(n_rows, n_cols)
        for i, spec in zip(range(n_features), gs):
            # determine which y-axes should be shared
            if sharey == 'all':
                cond = i != 0
            elif sharey == 'row':
                cond = i % n_cols != 0
            else:
                cond = False

            if cond:
                axes_ravel[i] = fig.add_subplot(spec, sharey=axes_ravel[i - 1])
                continue
            axes_ravel[i] = fig.add_subplot(spec)

    else:  # array-like
        if isinstance(ax, plt.Axes):
            ax = np.array(ax)
        if ax.size < n_features:
            raise ValueError("Expected ax to have {} axes, got {}".format(n_features, ax.size))
        axes = np.atleast_2d(ax)
        axes_ravel = axes.ravel()
        fig = axes_ravel[0].figure

    # make plots
    for ix, feature, ax_ravel in \
            zip(count(), features, axes_ravel):
        _ = _plot_one_ale_num(exp=exp,
                              feature=ix,
                              targets=targets,
                              constant=constant,
                              ax=ax_ravel,
                              legend=not ix,  # only one legend
                              line_kw=line_kw)

    # if explicit labels passed, handle the legend here as the axis passed might be repeated
    if line_kw['label'] is not None:
        axes_ravel[0].legend()

    fig.set(**fig_kw)
    # TODO: should we return just axes or ax + axes
    return axes

@no_type_check
def _plot_one_ale_num(exp: Explanation,
                      feature: int,
                      targets: TP.List[int],
                      constant: bool = False,
                      ax: 'plt.Axes' = None,
                      legend: bool = True,
                      line_kw: dict = None) -> 'plt.Axes':
    """
    Plots the ALE of exactly one feature on one axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if ax is None:
        ax = plt.gca()

    # add zero baseline
    ax.axhline(0, color='grey')

    # Sometimes within the computation of the ale values, we get more values corresponding
    # to the feature than values corresponding to the target or vice-versa, i.e. the number
    # of X's and Y's is not the same and therefore is not possible to properly build the ale
    # plot. These conditions help ensure len(x) equals len(y).
    if len(exp.feature_values[feature]) == len(exp.ale_values[feature][:, targets]):
        lines = ax.plot(
            exp.feature_values[feature],
            exp.ale_values[feature][:, targets] + constant * exp.constant_value,
            **line_kw
        )
    elif len(exp.feature_values[feature]) < len(exp.ale_values[feature][:, targets]):
        diff = len(exp.ale_values[feature][:, targets]) - len(exp.feature_values[feature])
        x = np.append(exp.feature_values[feature], np.repeat(exp.feature_values[feature][-1], diff))
        y = exp.ale_values[feature][:, targets]
        lines = ax.plot(
            x, y, **line_kw
        )
    elif len(exp.feature_values[feature]) > len(exp.ale_values[feature][:, targets]):
        diff = len(exp.feature_values[feature]) - len(exp.ale_values[feature][:, targets])
        y = np.append(exp.ale_values[feature][:, targets], np.repeat(exp.ale_values[feature][:, targets][-1], diff))
        x = exp.feature_values[feature]
        lines = ax.plot(
            x, y, **line_kw
        )

    # add decile markers to the bottom of the plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.feature_deciles[feature][1:], 0, 0.05, transform=trans)

    ax.set_xlabel(exp.feature_names[feature])
    ax.set_ylabel('ALE')

    if legend:
        # if no explicit labels passed, just use target names
        if line_kw['label'] is None:
            ax.legend(lines, exp.target_names[targets])

    return ax
