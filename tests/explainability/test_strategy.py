import unittest
import typing as TP
import functools

from mercury.explainability.explainers import cf_strategies as strat

import pytest
import numpy as np


class StrategyTests(unittest.TestCase):

    # The 2D-Rosenbrock function à la classifier output
    def _rosenbrock(self, x: 'np.ndarray', a: int=1, b: int=100) -> 'np.ndarray':
        assert x.shape[1] == 2, 'Invalid input for Rosenbrock function'

        # return a numpy array, instead of a scalar to simulate a
        # classifier "probability" (2nd field will be ignored)
        def _rosenbrock(x):
            return np.array([(a - x[0])**2 + b * (x[1] - x[0]**2)**2])

        return np.apply_along_axis(_rosenbrock, 1, x)
        # return np.array([[((a - x[:, 0])**2 + b * (x[:, 1] - x[:, 0]**2)**2)[0], 0.]])

    def _strategy_with(self, fn: TP.Callable[['np.ndarray'], 'np.ndarray'], strategy: str='backtracking') -> None:
        if strategy == 'backtracking':
            s1 = strat.Backtracking(self.x0, self.bounds, self.backtracking_step, fn, self.class_idx, self.threshold, self.kernel)
            x, p, visited, explored = s1.run(max_iter=self.max_iter)
            assert abs(x[0] - 1) <= self.diff and abs(x[1] - 1) <= self.diff
        else:
            s2 = strat.SimulatedAnnealing(self.x0, self.bounds, self.simanneal_step, fn, self.class_idx, self.threshold, self.kernel)  # type: strat.SimulatedAnnealing
            x, p, visited, _ = s2.run(tmax=20, tmin=1e-4, steps=5e4)
            assert abs(x[0] - 1) <= self.diff and abs(x[1] - 1) <= self.diff

    @pytest.fixture(autouse=True)
    def deterministic(self):
        np.random.seed(1)
        self.x0 = np.random.uniform(-3, 3, size=2)
        self.bounds = np.array([[-4, 4], [-4, 4]])
        self.backtracking_step = np.array([0.01, 0.01])
        self.simanneal_step = np.array([0.1, 0.1])
        self.class_idx = 0
        self.threshold = 0.
        self.kernel = np.ones(2)
        self.max_iter = 2000
        self.diff = 1e-2

    # @pytest.mark.skip(reason='Not finding global minima yet')
    def test_backtracking_rosenbrock(self) -> None:
        self._strategy_with(functools.partial(self._rosenbrock, a=1, b=100))    # type: ignore

    def test_simanneal_rosenbrock(self) -> None:
        self._strategy_with(functools.partial(self._rosenbrock, a=1, b=100), strategy='simanneal')      # type: ignore
