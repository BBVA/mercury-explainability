import numpy as np
import pandas as pd
import pytest
from mercury.explainability.explainers.monotonicity  import MonotonicityExplainer, nmi, pdp_curve


# ---------- nmi tests ----------

def test_nmi_length_mismatch():
    with pytest.raises(ValueError):
        nmi([1, 2, 3], [1, 2])


def test_nmi_strictly_increasing():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert nmi(x, y) == 0.0

def test_nmi_strictly_decreasing():
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]
    assert nmi(x, y) == 0.0

def test_nmi_constant_sequence():
    x = [1, 2, 3, 4]
    y = [5, 5, 5, 5]
    assert nmi(x, y) == 0.0


def test_nmi_max():
    x = [1, 2, 3, 4, 5]
    y = [2, 1, 2, 1, 1]
    assert nmi(x, y) == 1.0

def test_nmi_known_pattern():
    # y diffs: + + - - => 1 flip / (4-1) = 1/3
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 2, 1, 0]
    assert nmi(x, y) == pytest.approx(1 / 3)


# ---------- pdp_curve tests ----------

def test_pdp_curve_linear_mean_equals_median_dataframe():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "f1": rng.normal(size=50),
        "f2": rng.normal(size=50),
        "f3": rng.normal(size=50)
    })

    def pred_fn(X: pd.DataFrame):
        # Linear combination monotonic in f2
        return 3 * X["f2"].values + 1.5

    x_mean, y_mean = pdp_curve(pred_fn, df, "f2", n_grid=25, agg="mean")
    x_med, y_med = pdp_curve(pred_fn, df, "f2", n_grid=25, agg="median")

    assert len(x_mean) == 25
    assert np.allclose(x_mean, x_med)
    assert np.allclose(y_mean, y_med)


def test_pdp_curve_invalid_agg():
    X = np.random.randn(20, 3)

    def pred_fn(a):
        return a[:, 0] * 0 + 1

    with pytest.raises(ValueError):
        pdp_curve(pred_fn, X, 0, agg="wrong")


def test_pdp_curve_respects_sample_idx():
    X = np.random.randn(10, 4)
    sample_idx = [0, 2, 4]
    call_lengths = []

    def pred_fn(a):
        call_lengths.append(len(a))
        return np.ones(len(a))

    xg, yc = pdp_curve(pred_fn, X, 1, n_grid=15, sample_idx=sample_idx, agg="median")
    assert len(xg) == 15
    assert len(yc) == 15
    assert set(call_lengths) == {len(sample_idx)}
    assert np.allclose(yc, 1.0)


# ---------- MonotonicityExplainer tests ----------

def test_explainer_pdp_monotonic_numpy():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 5, size=(120, 3))

    def pred_fn(a):
        # Strictly increasing in feature 1
        return 2.0 * a[:, 1] + 0.7

    explainer = MonotonicityExplainer(pred_fn)
    explanation = explainer.explain(X, feature=1, strategy="pdp", n_grid=40, agg="mean")
    assert isinstance(explanation, dict)
    assert "nmi_value" in explanation
    assert explanation["feature"] == 1
    assert explanation["nmi_value"] == 0.0
    assert len(explanation["x_grid"]) == 40
    assert len(explanation["y_curve"]) == 40


def test_explainer_pdp_highly_non_monotonic():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(150, 2))

    def pred_fn(a):
        # Highly oscillatory in feature 0 to induce many flips
        return np.sin(100 * np.pi * a[:, 0])

    explainer = MonotonicityExplainer(pred_fn)
    explanation = explainer.explain(X, feature=0, strategy="pdp", n_grid=120, agg="median")
    assert isinstance(explanation, dict)
    assert explanation["feature"] == 0
    # Expect multiple direction changes -> moderate/high NMI
    assert explanation["nmi_value"] > 0.5
    assert explanation["nmi_value"] < 1.01


def test_explainer_invalid_strategy():
    X = np.random.randn(20, 2)

    def pred_fn(a):
        return a[:, 0]

    explainer = MonotonicityExplainer(pred_fn)
    with pytest.raises(ValueError):
        explainer.explain(X, feature=0, strategy="unknown")


def test_explainer_pdp_custom_grid():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))

    def pred_fn(a):
        return a[:, 2] ** 2  # still monotonic over non-negative subset? could be non-monotonic but fine

    grid = np.linspace(-2, 2, 31)
    explainer = MonotonicityExplainer(pred_fn)
    explanation = explainer.explain(X, feature=2, strategy="pdp", grid=grid, agg="mean")
    assert np.allclose(explanation["x_grid"], grid)
    assert len(explanation["y_curve"]) == len(grid)
    assert explanation["feature"] == 2


def test_explainer_wrong_inputs():

    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(150, 2))

    def pred_fn(a):
        return np.sin(100 * np.pi * a[:, 0])

    explainer = MonotonicityExplainer(pred_fn)
    with pytest.raises(ValueError):
        explanation = explainer.explain(X, feature="f1", strategy="pdp", n_grid=120, agg="median")

    with pytest.raises(ValueError):
        explanation = explainer.explain(X, feature=5, strategy="pdp", n_grid=120, agg="median")

    df = pd.DataFrame(X, columns=["f1", "f2"])
    with pytest.raises(ValueError):
        explanation = explainer.explain(df, feature=1, strategy="pdp", n_grid=120, agg="median")

    with pytest.raises(ValueError):
        explanation = explainer.explain(df, feature="f3", strategy="pdp", n_grid=120, agg="median")


def test_index_with_pandas_input():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(150, 2))
    df = pd.DataFrame(X, columns=["f1", "f2"])
    idx = [5, 10, 15, 20]

    def pred_fn(data):
        return data["f1"].values * 5.0

    explainer = MonotonicityExplainer(pred_fn)
    explanation = explainer.explain(df, feature="f1", sample_idx=idx)
    assert explanation["nmi_value"] >= 0.
    assert explanation["nmi_value"] <= 1.0


if __name__ == "__main__":
	import pytest
	pytest.main([__file__])
