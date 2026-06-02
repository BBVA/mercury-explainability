from typing import Union, Sequence, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from .explainer import MercuryExplainer


def nmi(x: Union[Sequence[float], np.ndarray], y: Union[Sequence[float], np.ndarray]) -> float:
    """
    Compute Non-Monotonic Index (NMI) for a unique feature.

    Args:
        x (Sequence[float], np.ndarray]): array-like of feature values
        y (Sequence[float], np.ndarray]): array-like of predictions corresponding to x

    Returns:
        A float in [0, 1] representing the Non-Monotonic Index (NMI).

    Example:
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 15, 25, 30]
        nmi_value = nmi(x, y)
        print(nmi_value)  # Output: NMI value
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x (Feature values) and y (predictions) must have same length")

    # sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]

    # finite differences
    dy = np.diff(y)

    # discard near-zero slopes to avoid noisy flips
    eps = np.finfo(float).eps * np.maximum(1.0, np.abs(y).max())
    signs = np.sign(np.where(np.abs(dy) <= eps, 0.0, dy))

    # remove zeros to count real direction changes only
    nz = signs[signs != 0]
    if len(nz) <= 1:
        return 0.0

    flips = np.sum(nz[1:] * nz[:-1] < 0)  # sign flips
    max_flips = len(nz) - 1               # normalization
    return flips / max_flips


def pdp_curve(
    f_pred: Callable[[Union[pd.DataFrame, np.ndarray]], np.ndarray],
    data: Union[pd.DataFrame, np.ndarray],
    feature: Union[str, int],
    grid: Optional[Sequence[float]] = None,
    n_grid: int = 50,
    agg: str = "median",
    sample_idx: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns x_grid, y_curve where y_curve is a PDP (or median ICE) over data for feature.
    Works for sklearn-style models with .predict.

    Args:
        f_pred (Callable[[Union[pd.DataFrame, np.ndarray]], np.ndarray]): prediction function (sklearn-style model with .predict)
        data (Union[pd.DataFrame, np.ndarray]): pandas DataFrame or numpy array with features
        feature (Union[str, int]): str (for DataFrame) or int (for array), feature name or index to analyze
        grid (Optional[Sequence[float]]): array-like, optional grid values. If None, creates linear grid
        n_grid (int): number of grid points if grid is None
        agg (str): str, 'mean' for standard PDP or 'median' for robust median ICE
        sample_idx (Optional[Sequence[int]]): array-like, indices of samples to use. If None, uses all samples

    Returns:
        x_grid: array of feature values
        y_curve: array of aggregated predictions

    Example:
        # For pandas DataFrame
        x_grid, y_curve = pdp_curve(model.predict, df, 'feature1', n_grid=30)

        # For numpy array
        x_grid, y_curve = pdp_curve(model.predict, X, 0, n_grid=30)
    """

    is_dataframe = isinstance(data, pd.DataFrame)

    if is_dataframe:
        x_col, base_data = _get_feature_data_df(data, feature, sample_idx)
    else:
        x_col, base_data = _get_feature_data_numpy(data, feature, sample_idx)

    # Create grid over the chosen feature
    if grid is None:
        x_min, x_max = np.min(x_col), np.max(x_col)
        x_grid = np.linspace(x_min, x_max, n_grid)
    else:
        x_grid = np.asarray(grid)

    preds = []

    for val in x_grid:
        if is_dataframe:
            data_mod = base_data.copy()
            data_mod[feature] = val
        else:
            data_mod = base_data.copy()
            data_mod[:, feature] = val

        y_hat = f_pred(data_mod)
        preds.append(y_hat)

    preds = np.vstack(preds)  # shape (n_grid, len(sample_idx))

    if agg == "mean":
        y_curve = np.mean(preds, axis=1)
    elif agg == "median":
        y_curve = np.median(preds, axis=1)
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    return x_grid, y_curve


def _get_feature_data_df(
    df: pd.DataFrame,
    feature: str,
    sample_idx: Optional[Union[Sequence[int], pd.Index]] = None
) -> Tuple[pd.Series, pd.DataFrame, int]:
    """
    Extract feature column and base data from DataFrame.
    """

    if not isinstance(feature, str):
        raise ValueError("For DataFrame input, feature must be a string (column name)")
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns")

    if sample_idx is None:
        sample_idx = df.index
    elif isinstance(sample_idx, (list, np.ndarray)):
        sample_idx = pd.Index(sample_idx)

    x_col = df[feature]
    base_data = df.loc[sample_idx].copy()

    return x_col, base_data


def _get_feature_data_numpy(
    data: np.ndarray,
    feature: int,
    sample_idx: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract feature column and base data from numpy array.
    """

    data = np.asarray(data)
    n, d = data.shape

    if not isinstance(feature, int):
        raise ValueError("For array input, feature must be an integer (column index)")
    if feature < 0 or feature >= d:
        raise ValueError(f"Feature index {feature} out of bounds for array with {d} columns")

    if sample_idx is None:
        sample_idx = np.arange(n)

    x_col = data[:, feature]
    base_data = data[sample_idx].copy()

    return x_col, base_data


class MonotonicityExplainer(MercuryExplainer):
    """
    Explainer for analyzing feature monotonicity in machine learning models.

    This explainer computes the Non-Monotonic Index (NMI) to quantify how much a feature
    violates monotonicity assumptions in model predictions. The NMI ranges from 0 to 1,
    where 0 indicates perfect monotonicity (predictions always increase or decrease with
    the feature) and 1 indicates maximum non-monotonicity (predictions change direction
    at every point).

    Currently, the explainer supports one strategy:
    - 'pdp': Uses Partial Dependence Plot curves with aggregation (mean/median) across samples


    Args:
        pred_fn (Callable[[Union[pd.DataFrame, np.ndarray]], np.ndarray]): model prediction function
            (like predict or predict_proba), Should accept pandas DataFrame or numpy arrays and return numpy arrays.

    Example:
        >>> # Numpy and classification
        >>> model_pred_fn = lambda X: model.predict_proba(X)[:, 1]
        >>> explainer = MonotonicityExplainer(model_pred_fn)
        >>> nmi_value = explainer.explain(X, feature=0)
        >>> print(f"Feature 0 monotonicity violation: {explanation.nmi_value}")

        >>> # Pandas DataFrame and regression
        >>> explainer = MonotonicityExplainer(model.predict)
        >>> nmi_value = explainer.explain(df, feature='feature_name')
        >>> print(f"NMI for 'feature_name': {explanation.nmi_value}")

    """

    def __init__(self, pred_fn: Callable[[Union[pd.DataFrame, np.ndarray]], np.ndarray]):
        super().__init__()
        self.pred_fn = pred_fn

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature: Union[str, int],
        strategy: str = "pdp",
        grid: Optional[Union[Sequence[float], np.ndarray]] = None,
        n_grid: int = 50,
        agg: str = "median",
        sample_idx: Optional[Union[Sequence[int], np.ndarray, pd.Index]] = None,
    ):
        """


        Compute the Non-Monotonic Index (NMI) for a specified feature using the chosen strategy.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): pandas DataFrame or numpy array with features
            feature (Union[str, int]): Feature to analyze - string (column name) for DataFrame or int (column index) for array
            strategy (str): Analysis strategy, only 'pdp' (Partial Dependence Plot) available currently
            grid (Optional[Union[Sequence[float], np.ndarray]]): Custom grid values for feature analysis.
                If None, creates linear grid from min to max feature values
            n_grid (int): int, number of grid points if grid is None
            agg (str): Aggregation method for PDP strategy - 'mean' or 'median'
            sample_idx (Optional[Union[Sequence[int], np.ndarray, pd.Index]]): array-like, indices of samples to use.
                If None, uses all samples

        Returns:
            (dict): Explanation object containing feature name, NMI value, x_grid, and y_curve

        Example:
            >>> explanation = explainer.explain(X, feature=0, strategy='pdp', n_grid=30)
            >>> print(f"NMI: {explanation.nmi_value}")
        """

        if strategy.lower() == "pdp":
            x_grid, y_curve = pdp_curve(
                self.pred_fn,
                X,
                feature,
                grid=grid,
                n_grid=n_grid,
                agg=agg,
                sample_idx=sample_idx
            )
        else:
            raise ValueError("invalid strategy")

        nmi_value = nmi(x_grid, y_curve)
        explanation = {'feature': feature, 'nmi_value': nmi_value, 'x_grid': x_grid, 'y_curve': y_curve}
        return explanation
