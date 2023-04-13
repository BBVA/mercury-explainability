"""
Optimization strategies for simple counterfactual explanations.
"""

import random
import typing as TP
from abc import ABCMeta, abstractmethod
import queue
import sys

from simanneal import Annealer
import numpy as np
import pandas as pd


class Strategy(metaclass=ABCMeta):
    """ Base class for explanation strategies. """

    def __init__(self,
                 state: 'np.ndarray',
                 bounds: 'np.ndarray',
                 step: 'np.ndarray',
                 fn: TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], 'np.ndarray'],
                 class_idx: int = 1,
                 threshold: float = 0.,
                 kernel: TP.Optional['np.ndarray'] = None,
                 report: bool = False) -> None:

        # Starting point
        self.state = state
        # Dimension bounds
        assert bounds.shape[0] == state.shape[0]
        self.bounds = bounds
        assert step.shape[0] == bounds.shape[0]
        self.step = step
        # Classifier fn
        self.fn = fn
        # Class
        self.class_idx = class_idx
        # Explored points
        self.explored = [self.state]
        # Probability threshold
        assert threshold >= 0. and threshold <= 1.
        self.threshold = threshold
        # Energies
        self.energies = [self.fn(self.state.reshape(1, -1))[:, self.class_idx][0]]
        # Kernel
        if kernel is None:
            kernel = np.ones(self.state.shape[0])
        # minimization or maximization problem?
        cur_prob = self.fn(self.state.reshape(1, -1))[0, self.class_idx]
        if cur_prob > self.threshold:
            self.min = True
        else:
            self.min = False

        self.kernel = kernel
        self.report = report

    @abstractmethod
    def run(self, *args, **kwargs) -> TP.Tuple['np.ndarray', float, 'np.ndarray', TP.Optional['np.ndarray']]:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass


class SimulatedAnnealing(Strategy, Annealer):
    """
    Simulated Annealing strategy.

    Args:
        state (np.ndarray):
            Initial state (initial starting point).
        bounds (np.ndarray):
            Bounds to be used when moving around the probability space defined by `fn`.
        step (np.ndarray):
            Step size values to be used when moving around the probability space defined by `fn`.
            Lower values may take more time/steps to find a solution while too large values may
            make impossible to find a solution.
        fn (TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], 'np.ndarray']):
            Classifier `predict_proba`- like function.
        class_idx (int):
            Class to be explained (e.g. 1 for binary classifiers). Default value is 1.
        threshold (float):
            Probability to be achieved (if path is found). Default value is 0.0.
        kernel (TP.Optional['np.ndarray']):
            Used to penalize certain dimensions when trying to move around the probability
            space (some dimensions may be more difficult to explain, hence don't move along them).
        report (bool):
            Whether to display probability updates during space search.
    """

    def __init__(self,
                 state: 'np.ndarray',
                 bounds: 'np.ndarray',
                 step: 'np.ndarray',
                 fn: TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], 'np.ndarray'],
                 class_idx: int = 1,
                 threshold: float = 0.,
                 kernel: TP.Optional['np.ndarray'] = None,
                 report: bool = False) -> None:
        Strategy.__init__(self, state, bounds, step, fn, class_idx, threshold, kernel, report)
        # Annealer's __init__ uses signal, see Annealers's init:
        # https://github.com/perrygeo/simanneal/blob/b2576eb75d88f8b8c91d959a44dd708706bb108e/simanneal/anneal.py#L52
        # This may break execution (depending on where simanneal runs, e.g. threads)
        # Overload it!
        self.state = Annealer.copy_state(self, state)

    def move(self) -> None:
        """ Move step in Simulated Annealing. """
        self.state = np.random.uniform(-0.1, 0.1, size=self.state.shape[0]) * self.kernel * self.step + self.state
        for i, bound in enumerate(self.bounds):
            if self.kernel[i] != 0:
                if self.state[i] < bound[0]:
                    self.state[i] = bound[0]
                elif self.state[i] > bound[1]:
                    self.state[i] = bound[1]
        self.explored.append(self.state)

    def energy(self) -> None:
        """ Energy step in Simulated Annealing. """
        p = self.fn(np.array(self.state.reshape(1, -1)))[0, self.class_idx]
        # energy check
        if self.min:
            if self.threshold is not None and p < self.threshold:
                p = self.threshold
        else:
            if self.threshold is not None and p > self.threshold:
                p = self.threshold
        self.energies.append(p)
        value = p if self.min else -p
        return value

    def best_solution(self, n: int = 3) -> 'np.ndarray':
        """ Returns the n best solutions found during the Simulated Annealing.

        Args:
            n (int): Number of solutions to be retrieved. Default value is 3.
        """

        ps = self.fn(np.array(self.explored))[:, self.class_idx]
        sorted_ps_idxs = np.argsort(ps)
        if not self.min:
            sorted_ps_idxs = sorted_ps_idxs[::-1]
        return np.array(self.explored)[sorted_ps_idxs[:n]]

    def run(self, *args, **kwargs) -> TP.Tuple['np.ndarray', float, 'np.ndarray', TP.Optional['np.ndarray']]:
        """ Kick off Simulated Annealing.

        Args:
            **kargs:
                Simulated Annealing specific arguments
                    - tmin: min temperature
                    - tmax: max temperature
                    - steps: number of iterations

        Returns:
            result (TP.Tuple['np.ndarray', float, 'np.ndarray', TP.Optional['np.ndarray']]):
                Tuple containing found solution, probability achieved, points visited and
                corresponding energies (i.e. probabilities).
        """

        if 'tmin' in kwargs:
            self.Tmin = kwargs['tmin']
        if 'tmax' in kwargs:
            self.Tmax = kwargs['tmax']
        if 'steps' in kwargs:
            self.steps = kwargs['steps']
            self.updates = self.steps
            # self.updates = 100
        sol, p = self.anneal()  # type: TP.Tuple['np.ndarray', float]
        return sol, p, np.array(self.explored), np.array(self.energies)

    def update(self, *args, **kwargs) -> None:
        # TODO: By default use default's simanneal update method if necessary
        if self.report:
            Annealer.update(self, *args, **kwargs)


class MyPriorityQueue(queue.PriorityQueue):

    def __init__(self):
        super().__init__()

    def get_same_priority(self, limit=None, block=False, shuffle_limit=False):
        e = self.get(block=block)
        elements = [e]
        prio = e[0]
        while self.qsize():
            e = self.get(block=block)
            if e[0] != prio:
                self.put(e)
                break
            elements.append(e)
            # if we already reached the limit and no shuffle, we stop here
            if (limit is not None) and (not shuffle_limit) and (len(elements) >= limit):
                return elements
        if limit is not None and len(elements) > limit:
            random.shuffle(elements)
            for e in elements[limit:]:
                self.put(e)
            return elements[:limit]
        return elements


class Backtracking(Strategy):
    """ 
    Backtracking strategy.

    Args:
        state ('np.ndarray'): Initial state (initial starting point).
        bounds (np.ndarray): Bounds to be used when moving around the probability space defined by `fn`.
        step (np.ndarray): Step size values to be used when moving around the probability space defined by `fn`.
            Lower values may take more time/steps to find a solution while too large values may
            make impossible to find a solution.
        fn (TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], 'np.ndarray']): Classifier `predict_proba`- like function.
        class_idx (int): Class to be explained (e.g. 1 for binary classifiers). Default value is 1.
        threshold (float): Probability to be achieved (if path is found). Default value is 0.0.
        kernel (TP.Optional['np.ndarray']): Used to penalize certain dimensions when trying to move around the probability
            space (some dimensions may be more difficult to explain, hence don't move along them).
        report (bool): Whether to display probability updates during space search.
        keep_explored_points (bool): Whether to keep the points that the algorithm explores. Setting it to False will
            decrease the computation time and memory usage in some cases. Default value is True.

    """

    def __init__(self,
                 state: 'np.ndarray',
                 bounds: 'np.ndarray',
                 step: 'np.ndarray',
                 fn: TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], 'np.ndarray'],
                 class_idx: int = 1,
                 threshold: float = 0.,
                 kernel: TP.Optional['np.ndarray'] = None,
                 report: bool = False,
                 keep_explored_points: bool = True,
        ) -> None:
        super().__init__(state, bounds, step, fn, class_idx, threshold, kernel, report)
        self.keep_explored_points = keep_explored_points

    def run(self, *args, **kwargs) -> TP.Tuple['np.ndarray', float, 'np.ndarray', TP.Optional['np.ndarray']]:
        """ Kick off the backtracking.

        Args:
            **kwargs: Backtracking specific arguments
                        - max_iter: max number of iterations

        Returns:
            result (TP.Tuple['np.ndarray', float, 'np.ndarray', TP.Optional['np.ndarray']]):
                Tuple containing found solution, probability achieved, points visited and
                corresponding energies (i.e. probabilities).
        """

        # Backtracking specific arguments
        if 'max_iter' in kwargs:
            max_iter = kwargs['max_iter']
        else:
            max_iter = 0
        if 'limit' in kwargs:
            limit = kwargs['limit']
        else:
            limit = None
        if 'shuffle_limit' in kwargs:
            shuffle_limit = kwargs['shuffle_limit']
        else:
            shuffle_limit = False

        # Backtracking starting point.
        point = self.state.reshape(1, -1)
        curr_prob = self.fn(point)[0, self.class_idx]
        best_point = point.copy()
        best_prob = curr_prob

        # Define condition (>= or <=) from current point and given threshold.
        # and the priority modifier (ascending or descending order) using prio_mod.
        if self.threshold > curr_prob:
            condition = float.__ge__
            prio_mod = lambda x: -x
        else:
            condition = float.__le__
            prio_mod = lambda x: x
            self.kernel = 1.1 - self.kernel

        visited = np.hstack([point, [[curr_prob]]])
        explored = np.hstack([point, [[curr_prob]]]) if self.keep_explored_points else np.array([])

        q = MyPriorityQueue()  # type: ignore
        # Avoid duplicates and querying known points.
        cache = {str(point.tolist()): curr_prob}
        points = [point]
        counter = 0
        while condition(self.threshold, curr_prob) and (not max_iter or counter < max_iter):
            # Explore neighbours.
            neighbours = np.zeros((0, self.step.size))
            neighbours_l = []
            neighbours_kernel = []
            local_cache = set()
            for p_ in points:
                for i, delta in enumerate(self.step):
                    # In every direction.
                    for sign in (-1, 1):
                        aux = p_.copy()
                        new_aux_i = aux[0, i] + sign * delta
                        if new_aux_i >= self.bounds[i][0] and new_aux_i <= self.bounds[i][1]:
                            aux[0, i] = new_aux_i
                            aux_l = aux.tolist()
                            str_aux_l = str(aux_l)
                            if str_aux_l not in cache and str_aux_l not in local_cache:
                                # If point is not in cache (visited) -> enqueue.
                                neighbours = np.vstack([neighbours, aux])
                                neighbours_l.append(aux_l)
                                neighbours_kernel.append(self.kernel[i])
                                local_cache.add(str_aux_l)
            if len(neighbours_l):
                assert neighbours.shape[0] == len(neighbours_kernel) == len(neighbours_l), \
                    'Number of neighbours should match.'
                probs = self.fn(neighbours)[:, self.class_idx]
                for n_idx, kernel, pt in zip(
                        range(probs.shape[0]), neighbours_kernel, neighbours_l):
                    prob = float(probs[n_idx])
                    prio = prob * kernel
                    q.put((prio_mod(prio), prob, pt))
                    if self.keep_explored_points:
                        explored = np.vstack(
                            [explored,
                            np.hstack([neighbours[n_idx].reshape(1, -1), [[prob]]])])
                    cache[str(pt)] = prob
            try:
                elements = q.get_same_priority(limit=limit, block=False, shuffle_limit=shuffle_limit)
                curr_prob = max([x[1] for x in elements])
                points = [np.array(x[2]) for x in elements]
            except queue.Empty:
                self.visited = visited
                self.explored = explored
                return best_point[0], best_prob, self.visited, self.explored
            if condition(curr_prob, best_prob):
                best_prob = curr_prob
                best_point = points[0]
            for point in points:
                visited = np.vstack([visited, np.hstack([point,
                                                         [[curr_prob]]])])

            if self.report:

                iter_string = f"{counter}/{max_iter}"
                update_string = f"\r{iter_string}\t\t{round(best_prob, 2)}"
                if counter == 0:
                    print('\rIteration\tBest Prob', file=sys.stderr)
                    print(update_string, file=sys.stderr, end="")
                else:
                    print(update_string, file=sys.stderr, end="")
                sys.stderr.flush()

            counter += 1 if max_iter else 0

        self.explored = explored
        self.visited = visited
        return best_point[0], best_prob, self.visited, self.explored

    def update(self, *args, **kwargs):
        if self.report:
            # TODO: An implementation for updates should be provided
            pass
