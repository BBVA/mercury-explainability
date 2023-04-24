import typing as TP
import numpy as np
import pandas as pd

from .explainer import MercuryExplainer
from .cf_strategies import SimulatedAnnealing, Backtracking

from mercury.explainability.explanations.counter_factual import (
    CounterfactualBasicExplanation
)


class CounterFactualExplainerBasic(MercuryExplainer):
    """
    Explains predictions on tabular (i.e. matrix) data for binary/multiclass classifiers.
    Currently two main strategies are implemented: one following a backtracking strategy and
    another following a probabilistic process (simulated annealing strategy).

    Args:
        train (TP.Union['np.ndarray', pd.DataFrame]):
            Training dataset to extract feature bounds from.
        fn (TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], TP.Union[float, 'np.ndarray']]):
            Classifier `predict_proba`-like function. Note that the returned probabilities
            must be valid, ie. the values must be between 0 and 1.
        labels (TP.List[str]):
            List of labels to be used when plotting results. If DataFrame used, labels take
            dataframe column names. Default is empty list.
        bounds (TP.Optional['np.ndarray']):
            Feature bounds used when no train data is provided (shape must match labels').
            Default is None.
        n_steps (int):
            Parameter used to indicate how small/large steps should be when exploring
            the space (default is 200).

    Raises:
        AssertionError:
            if bounds.size <= 0 when no train data is provided |
            if bounds.ndim != 2 when no train data is provided |
            if bounds.shape[1] != 2 when no train data is provided |
            if bounds.shape[0] != len(labels)
        TypeError:
            if train is not a DataFrame or numpy array.
    """

    def __init__(self,
                 train: TP.Union[TP.Optional['np.ndarray'], TP.Optional[pd.DataFrame]],
                 fn: TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], TP.Union[float, 'np.ndarray']],
                 labels: TP.List[str] = [],
                 bounds: TP.Optional['np.ndarray'] = None,
                 n_steps: int = 200) -> None:
        if train is None:
            # If data is not provided, labels and bounds are required
            assert bounds.size > 0, 'Bounds are required if no data is provided'
            assert bounds.ndim == 2 and bounds.shape[1] == 2, 'min/max values are required for each feature'
            assert len(labels) == bounds.shape[0], \
                'Labels and bound shapes must match, got {} and {} respectively' \
                    .format(len(labels), bounds.shape[0])
            # min/max values for each feature
            self.labels = labels
            self.bounds = bounds
        else:
            # Compute bounds
            if isinstance(train, pd.DataFrame):
                self.labels = train.columns.tolist()
                self.bounds = train.describe().loc[['min', 'max']].values.T
                assert len(self.labels) == self.bounds.shape[0], \
                    'Labels and bound shapes must match, got {} and {} respectively' \
                        .format(len(self.labels), self.bounds.shape[0])
            elif isinstance(train, np.ndarray):
                self.labels = labels
                self.bounds = np.stack([
                    np.apply_along_axis(np.min, 0, train),
                    np.apply_along_axis(np.max, 0, train)], axis=1)
                assert len(self.labels) == self.bounds.shape[0], \
                    'Labels and bound shapes must match, got {} and {} respectively' \
                        .format(len(self.labels), self.bounds.shape[0])
            else:
                raise TypeError('Invalid type for argument train, got {} but expected numpy array or pandas dataframe'.
                                format(type(train)))

        # Compute steps
        self.n_steps = n_steps
        self.step = (self.bounds[:, 1] - self.bounds[:, 0]) / self.n_steps

        # Function to be evaluated on optimization
        self.fn = fn

    def explain(self,
                from_: 'np.ndarray',
                threshold: float,
                class_idx: int = 1,
                kernel: TP.Optional['np.ndarray'] = None,
                bounds: TP.Optional['np.ndarray'] = None,
                step: TP.Optional['np.ndarray'] = None,
                strategy: str = 'backtracking',
                report: bool = False,
                keep_explored_points: bool = True,
                **kwargs) -> CounterfactualBasicExplanation:
        """
        Roll the panellet down the valley and find an explanation.

        Args:
            from_ ('np.ndarray'):
                Starting point.
            threshold (float):
                Probability to be achieved (if path is found).
            class_idx (int):
                Class to be explained (e.g. 1 for binary classifiers).
            kernel (TP.Optional['np.ndarray']):
                Used to penalize certain dimensions when trying to move around
                the probability space (some dimensions may be more difficult to explain,
                hence don't move along them). Default is np.ones(n), meaning all dimensions
                can be used to move around the space (must be a value between 0 and 1).
            bounds (TP.Optional['np.ndarray']):
                Feature bound values to be used when exploring the probability space. If not
                specified, the ones extracted from the training data are used instead.
            step (TP.Optional['np.ndarray']):
                Step values to be used when moving around the probability space. If not specified,
                training bounds are divided by 200 (arbitrary value) and these are used as step value.
            strategy (str):
                If 'backtracking', the backtracking strategy is used to move around the probability space.
                If 'simanneal', the simulated annealing strategy is used to move around the probability space.
            report (bool):
                Whether to report the algorithm progress during the execution.
            keep_explored_points (bool):
                Whether to keep the points that the algorithm explores. Setting it to False will decrease
                the computation time and memory usage in some cases. Default value is True.

        Raises:
            AssertionError:
                If `from_` number of dimensions is != 1 |
                If `from_` shape does not match `bounds` shape |
                If `bounds` shape is not valid |
                If `step` shape does not match `bounds` shape |
            ValueError:
                if strategy is not 'backtacking' or 'simanneal'.

        Returns:
            explanation (CounterfactualBasicExplanation):
                CounterfactualBasicExplanation with the solution found and how it differs from the starting point.
        """

        if kernel is None:
            self.kernel = np.ones(from_.shape[0])
        else:
            self.kernel = kernel

        assert from_.ndim == 1, \
            'Invalid starting point shape, got {} but expected unidimensional vector'.format(from_.shape)

        if bounds is not None:
            assert from_.shape[0] == bounds.shape[0], \
                'Starting point and bounds shapes should match, got {} and {}'.format(from_.shape, bounds.shape[0])

        # Update bounds based on reference point
        l_bounds = self.bounds.copy()
        for i, bound in enumerate(l_bounds):
            new_min = bound[0]
            new_max = bound[1]
            if from_[i] < bound[0]:
                new_min = from_[i]
            elif from_[i] > bound[1]:
                new_max = from_[1]
            l_bounds[i, 0] = new_min
            l_bounds[i, 1] = new_max

        # Update bounds, if new_bounds are specified
        if bounds is not None:
            assert bounds.shape == l_bounds.shape, \
                'Invalid dimensions for new bounds, got {} but expected {}'.format(bounds.shape, l_bounds.shape)
            for i, bound in enumerate(bounds):
                # Update bound only if starting point is within it in this dimension
                if bound[0] <= from_[i] and bound[1] >= from_[i]:
                    l_bounds[i] = bounds[i]

        if step is not None:
            assert step.shape[0] == l_bounds.shape[0], \
                'Invalid step shape, got {} but expected {}'.format(step.shape, l_bounds.shape[0])
            self.step = step

        if strategy == 'backtracking':
            # Backtracking strategy
            sol, p, visited, explored = Backtracking(from_, l_bounds, self.step, self.fn, class_idx,
                                                           threshold=threshold, kernel=self.kernel, report=report,
                                                           keep_explored_points=keep_explored_points).run(
                **kwargs)
            ps = visited[:, -1]
            visited = visited[:, :-1]
            if keep_explored_points:
                explored_points = explored[:, :-1]
                explored_ps = explored[:, -1]
            else:
                explored_points = np.array([])
                explored_ps = np.array([])
            return CounterfactualBasicExplanation(
                from_, sol, p, visited, ps, l_bounds, explored_points,
                explored_ps, labels=self.labels)
        elif strategy == 'simanneal':
            # Simulated Annealing strategy
            sol, p, visited, energies = SimulatedAnnealing(from_, l_bounds, self.step, self.fn, class_idx,
                                                                 threshold=threshold, kernel=self.kernel,
                                                                 report=report).run(**kwargs)
            return CounterfactualBasicExplanation(from_, sol, abs(p), visited, energies[:-1], l_bounds,
                                                   labels=self.labels)
        else:
            raise ValueError('Invalid strategy')
