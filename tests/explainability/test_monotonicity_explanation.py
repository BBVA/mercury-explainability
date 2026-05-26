from mercury.explainability.explanations.monotonicity import MonotonicityExplanation
import numpy as np
from unittest.mock import patch


def test_init_basic():
    explanation = MonotonicityExplanation({
        "feature": "feature1",
        "nmi_value": 0.85
    })
    assert explanation.feature == "feature1"
    assert explanation.nmi_value == 0.85


def test_init_with_all_parameters():
    x_grid = [1, 2, 3, 4, 5]
    y_curve = [0.1, 0.3, 0.5, 0.7, 0.9]

    explanation = MonotonicityExplanation({
        "feature": "temperature",
        "nmi_value": 0.75,
        "x_grid": x_grid,
        "y_curve": y_curve
    })
    assert explanation.feature == "temperature"
    assert explanation.nmi_value == 0.75
    assert explanation.x_grid == x_grid
    assert explanation.y_curve == y_curve


def test_init_with_numpy_arrays():
    x_grid = np.array([1, 2, 3, 4, 5])
    y_curve = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    explanation = MonotonicityExplanation({
        "feature": "age",
        "nmi_value": -0.25,
        "x_grid": x_grid,
        "y_curve": y_curve
    })

    assert explanation.feature == "age"
    assert explanation.nmi_value == -0.25
    np.testing.assert_array_equal(explanation.x_grid, x_grid)
    np.testing.assert_array_equal(explanation.y_curve, y_curve)

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.plot')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.grid')
@patch('matplotlib.pyplot.show')
def test_plot_default_parameters(mock_show, mock_grid, mock_ylabel,
                               mock_xlabel, mock_title, mock_plot, mock_figure):
    """Test plot method with default parameters."""
    x_grid = [1, 2, 3, 4, 5]
    y_curve = [0.1, 0.3, 0.5, 0.7, 0.9]
    explanation = MonotonicityExplanation({
        "feature": "feature1",
        "nmi_value": 0.85,
        "x_grid": x_grid,
        "y_curve": y_curve
    })

    explanation.plot()

    mock_figure.assert_called_once()
    mock_plot.assert_called_once()
    mock_xlabel.assert_called_once_with('feature1')
    mock_ylabel.assert_called_once()
    mock_grid.assert_called_once()
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.plot')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.grid')
@patch('matplotlib.pyplot.show')
def test_plot_custom_parameters(mock_show, mock_grid, mock_ylabel,
                              mock_xlabel, mock_title, mock_plot, mock_figure):
    """Test plot method with custom parameters."""
    x_grid = np.array([10, 20, 30])
    y_curve = np.array([0.2, 0.6, 0.8])
    explanation = MonotonicityExplanation({
        "feature": "custom_feature",
        "nmi_value": 0.42,
        "x_grid": x_grid,
        "y_curve": y_curve
    })

    explanation.plot(
        figsize=(10, 6),
        marker='s',
        title='Custom Title',
        xlabel='Custom X Label',
        ylabel='Custom Y Label',
        grid=False
    )

    mock_figure.assert_called_once_with(figsize=(10, 6))
    mock_plot.assert_called_once_with(x_grid, y_curve, marker='s')
    mock_title.assert_called_once_with('Custom Title')
    mock_xlabel.assert_called_once_with('Custom X Label')
    mock_ylabel.assert_called_once_with('Custom Y Label')
    mock_grid.assert_not_called()
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.plot')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.grid')
@patch('matplotlib.pyplot.show')
def test_plot_with_numpy_arrays(mock_show, mock_grid, mock_ylabel,
                               mock_xlabel, mock_title, mock_plot, mock_figure):
    """Test plot method with numpy arrays."""
    x_grid = np.linspace(0, 10, 5)
    y_curve = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    explanation = MonotonicityExplanation({
        "feature": "numpy_feature",
        "nmi_value": 0.88,
        "x_grid": x_grid,
        "y_curve": y_curve
    })

    explanation.plot()

    mock_plot.assert_called_once()


if __name__ == "__main__":
	import pytest
	pytest.main([__file__])
