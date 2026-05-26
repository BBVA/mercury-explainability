import unittest
import typing as TP
import functools
import queue

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


def _linear_prob_fn(x: 'np.ndarray') -> 'np.ndarray':
    """Simple 2-class score where class-0 probability increases with x0+x1."""
    score = x[:, 0] + x[:, 1]
    p0 = np.clip((score + 2.0) / 4.0, 0.0, 1.0)
    p1 = 1.0 - p0
    return np.vstack([p0, p1]).T


class _DummyStrategy(strat.Strategy):

    def run(self, *args, **kwargs):
        return strat.Strategy.run(self, *args, **kwargs)

    def update(self, *args, **kwargs):
        return strat.Strategy.update(self, *args, **kwargs)


def test_strategy_default_kernel_and_abstract_base_lines():
    state = np.array([0.0, 0.0])
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    step = np.array([0.1, 0.1])
    s = _DummyStrategy(state, bounds, step, _linear_prob_fn, class_idx=0, threshold=0.9, kernel=None)

    assert (s.kernel == np.ones(2)).all()
    assert s.min is False
    assert s.run() is None
    assert s.update() is None


def test_simanneal_move_clips_to_bounds(monkeypatch):
    state = np.array([0.0, 0.0])
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    step = np.array([20.0, 20.0])
    s = strat.SimulatedAnnealing(state, bounds, step, _linear_prob_fn, class_idx=0, threshold=0.0)

    monkeypatch.setattr(np.random, 'uniform', lambda low, high, size: np.array([-0.1, 0.1]))
    s.move()

    assert s.state[0] == -1.0
    assert s.state[1] == 1.0


def test_simanneal_energy_threshold_clip_for_min_and_max():
    state = np.array([0.0, 0.0])
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    step = np.array([0.1, 0.1])

    s_min = strat.SimulatedAnnealing(state, bounds, step, _linear_prob_fn, class_idx=0, threshold=0.2)
    s_min.state = np.array([-1.0, -1.0])
    energy_min = s_min.energy()
    assert energy_min == pytest.approx(0.2)

    s_max = strat.SimulatedAnnealing(state, bounds, step, _linear_prob_fn, class_idx=0, threshold=0.8)
    s_max.state = np.array([1.0, 1.0])
    energy_max = s_max.energy()
    assert energy_max == pytest.approx(-0.8)


def test_simanneal_best_solution_for_max_and_min_and_update_report(monkeypatch):
    state = np.array([0.0, 0.0])
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    step = np.array([0.1, 0.1])

    s_max = strat.SimulatedAnnealing(state, bounds, step, _linear_prob_fn, class_idx=0, threshold=0.8)
    s_max.explored = [np.array([-1.0, -1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    best_max = s_max.best_solution(n=1)
    assert np.allclose(best_max[0], np.array([1.0, 1.0]))

    s_min = strat.SimulatedAnnealing(state, bounds, step, _linear_prob_fn, class_idx=0, threshold=0.0)
    s_min.explored = [np.array([-1.0, -1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    best_min = s_min.best_solution(n=1)
    assert np.allclose(best_min[0], np.array([-1.0, -1.0]))

    called = {'value': False}

    def _fake_update(self, *args, **kwargs):
        called['value'] = True

    monkeypatch.setattr(strat.Annealer, 'update', _fake_update)
    s_report = strat.SimulatedAnnealing(state, bounds, step, _linear_prob_fn, class_idx=0, threshold=0.0, report=True)
    s_report.update(0, 0, 0, 0)
    assert called['value'] is True


def test_priority_queue_limit_and_shuffle_paths(monkeypatch):
    q = strat.MyPriorityQueue()
    q.put((1.0, 0.8, [1]))
    q.put((1.0, 0.7, [2]))
    q.put((1.0, 0.6, [3]))
    q.put((2.0, 0.5, [4]))

    limited = q.get_same_priority(limit=2, shuffle_limit=False)
    assert len(limited) == 2

    q2 = strat.MyPriorityQueue()
    q2.put((1.0, 0.8, [1]))
    q2.put((1.0, 0.7, [2]))
    q2.put((1.0, 0.6, [3]))
    q2.put((2.0, 0.5, [4]))

    monkeypatch.setattr(strat.random, 'shuffle', lambda items: items.reverse())
    shuffled_limited = q2.get_same_priority(limit=2, shuffle_limit=True)
    assert len(shuffled_limited) == 2
    assert q2.qsize() >= 1


def test_backtracking_threshold_greater_early_empty_queue_and_update_report(capsys):
    state = np.array([0.0, 0.0])
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    # Zero step means no neighbors are generated and queue.Empty branch is taken.
    step = np.array([0.0, 0.0])
    b = strat.Backtracking(
        state,
        bounds,
        step,
        _linear_prob_fn,
        class_idx=0,
        threshold=0.9,
        kernel=np.array([1.0, 1.0]),
        report=True,
        keep_explored_points=False,
    )

    best_point, best_prob, visited, explored = b.run(limit=1, shuffle_limit=True)

    assert np.allclose(best_point, state)
    assert best_prob == pytest.approx(_linear_prob_fn(state.reshape(1, -1))[0, 0])
    assert visited.shape[1] == 3
    assert explored.size == 0

    b.update()
    captured = capsys.readouterr()
    assert captured.err == ''


def test_backtracking_report_output_and_default_max_iter(capsys):
    state = np.array([0.0, 0.0])
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    step = np.array([0.5, 0.5])
    b = strat.Backtracking(
        state,
        bounds,
        step,
        _linear_prob_fn,
        class_idx=0,
        threshold=0.0,
        kernel=np.array([1.0, 1.0]),
        report=True,
    )

    _, _, visited, _ = b.run(max_iter=2)
    assert visited.shape[0] >= 1
    captured = capsys.readouterr()
    assert 'Iteration\tBest Prob' in captured.err

    # Cover default max_iter path (max_iter not passed).
    b2 = strat.Backtracking(
        state,
        bounds,
        step,
        _linear_prob_fn,
        class_idx=0,
        threshold=0.0,
        kernel=np.array([1.0, 1.0]),
        report=False,
    )
    b2.run()


if __name__ == "__main__":
	pytest.main([__file__])
