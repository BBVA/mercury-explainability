import unittest

from mercury.explainability.explainers import CounterFactualExplainerBasic

import numpy as np
import pandas as pd
import pytest


class CounterfactualBasicTest(unittest.TestCase):
    """Simple checks on Panellet."""

    # The 2D-Rosenbrock function à la classifier output
    def _rosenbrock(self, x: "np.ndarray", a: int = 1, b: int = 100) -> "np.ndarray":
        assert x.shape[1] == 2, "Invalid input for Rosenbrock function"

        # return a numpy array, instead of a scalar to simulate a
        # classifier "probability" (2nd field will be ignored)
        def _clip_rosenbrock(x):
            return np.clip(np.array([(a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2]), 0, 1)

        return np.apply_along_axis(_clip_rosenbrock, 1, x)
        # return np.array([[((a - x[:, 0])**2 + b * (x[:, 1] - x[:, 0]**2)**2)[0], 0.]])

    def test_nodata_constructor_invalid_bounds1(self):
        with self.assertRaises(AssertionError):
            # Labels/bounds shape does not match
            CounterFactualExplainerBasic(
                None, self._rosenbrock, labels=["a", "b"], bounds=np.array([])
            )

    def test_nodata_constructor_invalid_bounds2(self):
        with self.assertRaises(AssertionError):
            # Invalid bounds shape
            CounterFactualExplainerBasic(
                None, self._rosenbrock, labels=["a"], bounds=np.array([0])
            )

    def test_nodata_constructor_invalid_labels(self):
        with self.assertRaises(AssertionError):
            # Labels/bounds shape does not match
            CounterFactualExplainerBasic(
                None, self._rosenbrock, labels=["a"], bounds=np.array([[0, 1], [0, 1]])
            )

    def test_nodata_roll_invalid_from(self):
        pan = CounterFactualExplainerBasic(
            None, self._rosenbrock, labels=["a"], bounds=np.array([[0, 1]])
        )
        with self.assertRaises(AssertionError):
            pan.explain(np.array([[1, 2]]), 0.1)

    def test_nodata_roll_invalid_bounds(self):
        pan = CounterFactualExplainerBasic(
            None, self._rosenbrock, labels=["a"], bounds=np.array([[0, 1]])
        )
        with self.assertRaises(AssertionError):
            pan.explain(np.array([1]), 0.1, bounds=np.array([[0, 1], [0, 1]]))

    def test_nodata_roll_invalid_step(self):
        pan = CounterFactualExplainerBasic(
            None, self._rosenbrock, labels=["a"], bounds=np.array([[0, 1]])
        )
        with self.assertRaises(AssertionError):
            pan.explain(np.array([1]), 0.1, step=np.array([0.01, 0.01]))

    def test_nodata_bt_strategy(self):
        df_data = pd.DataFrame(data={"a": [-10, 10], "b": [-10, 10]})
        np_data = np.array([[-10, -10], [10, 10]])
        pan = CounterFactualExplainerBasic(
            np_data, self._rosenbrock, labels=["a", "b"]
        )
        pan.explain(np_data[0], 0.1, class_idx=0, strategy="backtracking")

    def test_build_wrong_dtype(self):
        with pytest.raises(TypeError) as excinfo:
            pan = CounterFactualExplainerBasic(
                [0,1], self._rosenbrock, labels=["a", "b"]
            )

    def test_nodata_sima_strategy(self):
        df_data = pd.DataFrame(data={"a": [-10, 10], "b": [-10, 10]})
        np_data = np.array([[-10, -10], [10, 10]])
        pan = CounterFactualExplainerBasic(
            df_data, self._rosenbrock
        )
        pan.explain(np_data[0], 0.1, 
            class_idx=0, 
            bounds=np.array([[0, 9], [0, 9]]),
            strategy="simanneal")

    def test_nodata_invalid_strategy(self):
        pan = CounterFactualExplainerBasic(
            None, self._rosenbrock, labels=["a"], bounds=np.array([[0, 1]])
        )
        with self.assertRaises(ValueError):
            pan.explain(np.array([1]), 0.1, strategy="meh")

    def test_data_constructor_invalid_labels(self):
        df_data = pd.DataFrame(data={"a": [-10, 10], "b": [-10, 10]})
        np_data = np.array([[-10, -10], [10, 10]])
        # this shold give no error
        CounterFactualExplainerBasic(df_data, self._rosenbrock)
        with self.assertRaises(AssertionError):
            # Labels/bounds shape does not match
            CounterFactualExplainerBasic(np_data, self._rosenbrock, labels=["a", "b", "c"])

    def test_keep_explored_points_false(self):
        np_data = np.array([[-10, -10], [10, 10]])
        pan = CounterFactualExplainerBasic(
            np_data, self._rosenbrock, labels=["a", "b"]
        )
        explanation = pan.explain(
            np_data[0], 0.1, class_idx=0, strategy="backtracking", limit=2, max_iter=3, keep_explored_points=False
        )
        assert len(explanation.explored) == 0

    def test_with_kernel_and_step(self):
        np_data = np.array([[-10, -10], [10, 10]])
        pan = CounterFactualExplainerBasic(
            np_data, self._rosenbrock, labels=["a", "b"],
        )
        explanation = pan.explain(
            np_data[0], 0.1, class_idx=0, strategy="backtracking", limit=1, max_iter=5, keep_explored_points=False,
            shuffle_limit=True,
            kernel=np.array([1.0, 0.]), step=np.array([0.2, 0.])
        )
        assert (pan.kernel == np.array([1.0, 0.])).all()
        assert (pan.step == np.array([0.2, 0.])).all()
        assert explanation.get_changes()[1] == 0
