import typing as TP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bokeh.plotting as BP
import bokeh.io as BPIO

from bokeh.models import ColorBar, LinearColorMapper
from bokeh.layouts import layout, row, column


class CounterfactualBasicExplanation(object):
    """
    A Panallet explanation.

    Args:
        from_ (np.ndarray):
            Starting point.
        to_ (np.ndarray):
            Found solution.
        p (float):
            Probability of found solution.
        path (np.ndarray):
            Path followed to get to the found solution.
        path_ps (np.ndarray):
            Probabilities of each path step.
        bounds (np.ndarray):
            Feature bounds used when exploring the probability space.
        explored (np.ndarray):
            Points explored but not visited (available only when backtracking
            strategy is used, empty for Simulated Annealing)
        explored_ps (np.ndarray):
            Probabilities of explored points (available only when backtracking
            strategy is used, empty for Simulated Annealing)
        labels (TP.Optional[TP.List[str]]):
            Labels to be used for each point dimension (used when plotting).

    Raises:
        AssertionError: if from_ shape != to_.shape
        AssertionError: if dim(from_) != 1
        AssertionError: if not 0 <= p <= 1
        AssertionError: if path.shape[0] != path_ps.shape[0]
        AssertionError: if bounds.shape[0] != from_.shape[0]
        AssertionError: if explored.shape[0] != explored_ps.shape[0]
        AssertionError: if len(labels) > 0 and len(labels) != bounds.shape[0]
    """
    def __init__(self,
                 from_: 'np.ndarray',
                 to_: 'np.ndarray',
                 p: float,
                 path: 'np.ndarray',
                 path_ps: 'np.ndarray',
                 bounds: 'np.ndarray',
                 explored: 'np.ndarray' = np.array([]),
                 explored_ps: 'np.ndarray' = np.array([]),
                 labels: TP.Optional[TP.List[str]] = []) -> None:
        # Initial/end points
        assert from_.shape == to_.shape and from_.ndim == 1, 'Invalid dimensions'
        self.from_ = from_
        self.to_ = to_

        # Found solution probability
        assert p >= 0 and p <= 1, 'Invalid probability'
        self.p = p

        # Path followed till solution is found
        assert path.shape[0] == path_ps.shape[0], \
            'Invalid shape for path probabilities, got {} but expected {}'.format(path.shape[0], path_ps.shape[0])
        self.path = path
        self.path_ps = path_ps

        # Used bounds in the solution
        assert bounds.shape[0] == self.from_.shape[0], 'Invalid bounds shape'
        self.bounds = bounds

        assert explored.shape[0] == explored_ps.shape[0], \
            'Invalid shape for explored probabilities, got {} but expected {}'.format(explored.shape[0],
                                                                                      explored_ps.shape[0])
        self.explored = explored

        if labels is not None and len(labels) > 0:
            assert len(labels) == self.bounds.shape[0], 'Invalid number of labels'
        self.labels = labels

    def get_changes(self, relative=True) -> 'np.ndarray':
        """
        Returns relative/absolute changes between initial and ending point.

        Args:
            relative (bool):
                True for relative changes, False for absolute changes.

        Returns:
            (np.ndarray) Relative or absolute changes for each feature.
        """

        if relative:
            # Avoid divs by zero
            aux = self.from_.copy()
            aux[aux == 0.] = 1.
            return (self.to_.squeeze() - self.from_.squeeze()) * 100 / (np.sign(aux) * aux.squeeze())
        else:
            return self.to_.squeeze() - self.from_.squeeze()

    @staticmethod
    def plot_butterfly(data,
                       columns,
                       axis,
                       title: str = '',
                       decimals: int = 1,
                       positive_color: str = '#0A5FB4',
                       negative_color: str = '#DA3851',
                       special_color: str = '#48AE64') -> None:  # pragma: no cover
        data_ = data.copy().squeeze()
        num_labels = [(' {:.' + str(decimals) + 'f} ').format(float(x)) for x in data_]
        are_normal = np.isfinite(data_)
        are_special = False == are_normal
        if sum(are_normal):
            imputed_pos = max(0., np.max(data_[are_normal]))
            imputed_neg = min(0., np.min(data_[are_normal]))
        else:
            imputed_pos, imputed_neg = 1., -1.
        data_[np.logical_and(are_special, np.isnan(data_))] = 0.
        data_[np.logical_and(are_special, np.isposinf(data_))] = imputed_pos
        data_[np.logical_and(are_special, np.isneginf(data_))] = imputed_neg
        rec = axis.barh(range(data_.size), data_)
        axis.tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')
        for i, (value, label, num_label, is_normal) in enumerate(zip(data_, columns, num_labels, are_normal)):
            if is_normal:
                color = negative_color if value < 0 else positive_color
            else:
                color = special_color
            num_align, label_align = ('left', 'right') if value > 0 else ('right', 'left')
            axis.text(value, i, num_label, ha=num_align, va='center', color=color, size='smaller')
            axis.text(0, i, ' {} '.format(label), ha=label_align, va='center', color=color, size='smaller')
            rec[i].set_color(color)
        axis.set_axis_off()
        axis.set_title(title)
        axis.title.set_text(title)

    def show(self, figsize: TP.Tuple[int, int] = (12, 6), debug: bool = False,
             path: TP.Optional[str] = None, backend='matplotlib') -> None:  # pragma: no cover
        """
        Creates a plot with the explanation.

        Args:
            figsize (tuple):
                Width and height of the figure (inches if matplotlib backend is used,
                pixels for bokeh backend).
            debug (bool):
                Display verbose information (debug mode).
        """

        def _show(from_: 'np.ndarray', to_: 'np.ndarray', backend='matplotlib',
                  path: TP.Optional[str] = None, debug: bool = False) -> None:
            """ Backend specific show method. """

            if backend == 'matplotlib':
                # It seems we can't decouple figure from axes
                fig = plt.figure(figsize=figsize)

                # LIME-like hbars showing relative differences
                ax = plt.subplot2grid((2, 5), (0, 1))
                CounterfactualBasicExplanation.plot_butterfly(
                    self.get_changes(relative=False), self.labels, ax,
                    title='Absolute delta')

                ax = plt.subplot2grid((2, 5), (0, 3))
                CounterfactualBasicExplanation.plot_butterfly(
                    self.get_changes(relative=True), self.labels, ax,
                    title='Relative delta')

                # Probabilities
                ax = plt.subplot2grid((2, 5), (1, 0), colspan=5)
                xs = np.arange(len(self.path_ps))
                ys = self.path_ps
                cax = ax.scatter(xs, ys, c=ys)
                fig.colorbar(cax)
                ax.plot(xs, ys, '--', c='k', linewidth=.2, alpha=.3)
                ax.grid()
                ax.set_title('Visited itinerary')
                ax.set_xlabel('# Iteration')
                ax.set_ylabel('probability')
                plt.tight_layout()

                if path is not None:
                    plt.savefig(path, output='pdf')
                else:
                    plt.show()

            elif backend == 'bokeh':
                # LIME-like hbars showing relative differences
                values = self.get_changes()
                fig1 = BP.figure(plot_width=400, plot_height=300, y_range=self.labels,
                                 x_range=(min(values), max(values)), x_axis_label='Relative change')
                colors = np.where(values <= 0, '#ff0000', '#00ff00')
                fig1.hbar(y=self.labels, height=0.75, right=values, fill_color=colors)

                # LIME-like hbars showing absolute differences
                values = self.get_changes(relative=False)
                fig2 = BP.figure(plot_width=400, plot_height=300, y_range=self.labels,
                                 x_range=(min(values), max(values)), x_axis_label='Absolute change')
                colors = np.where(values <= 0, '#ff0000', '#00ff00')
                fig2.hbar(y=self.labels, height=0.75, right=values, fill_color=colors, line_color=None)

                # Probabilities
                fig3 = BP.figure(plot_width=800, plot_height=200, x_axis_label='Step', y_axis_label='p')
                xs = np.arange(self.path_ps.size)
                ys = self.path_ps
                color_mapper = LinearColorMapper(palette='Viridis256', low=min(self.path_ps),
                                                 high=max(self.path_ps))
                color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
                fig3.circle(xs, ys, size=5, fill_color={'field': 'y', 'transform': color_mapper},
                            fill_alpha=.3, line_color=None)
                fig3.add_layout(color_bar, 'left')
                fig3.line(xs, ys, line_dash='dashed', line_alpha=.3, line_width=.2)

                row1 = row([fig1, fig2])
                row2 = row([fig3])
                lyt = column([row1, row2])

                if path is not None:
                    BPIO.export_png(lyt, filename=path)
                else:
                    BPIO.output_notebook(hide_banner=True)
                    BP.show(lyt)

            else:
                raise ValueError('Unsupported backend')

            if debug:
                self.__verbose()

        _show(self.from_, self.to_, debug=debug, path=path, backend=backend)

    def __verbose(self):  # pragma: no cover
        """ Internal debug information. """

        print('Used bounds:')
        for i in range(self.bounds.shape[0]):
            label = self.labels[i] if self.labels else ''
            print('\t[{}] {}: [{}, {}]'.format(i, label, self.bounds[i][0], self.bounds[i][1]))
        print('Starting point: {}'.format(self.from_))
        print('Found solution: {} with probability {}'.format(self.to_, self.p))
        print('Changes:')
        for i in range(self.from_.shape[0]):
            if self.from_[i] != self.to_[i]:
                label = self.labels[i] if self.labels else ''
                print('\t[{}] {}: {} -> {}'.format(i, label, self.from_[i], self.to_[i]))


class CounterfactualWithImportanceExplanation(object):
    """
    Extended Counterfactual Explanations

    Args:
        explain_data:
            A pandas DataFrame containing the observations for which an explanation has to be found.
        explanations:
            A list containing the results of computing the explanations for explain_data.
        categorical:
            A dictionary containing as key the features that are categorical and as value, the possible
            categorical values.
    """

    def __init__(
            self,
            explain_data: pd.DataFrame,
            counterfactuals: TP.List[dict],
            importances: TP.List[TP.Tuple],
            count_diffs: dict,
            count_diffs_norm: dict
        ) -> None:
        self.explain_data = explain_data
        self.counterfactuals = counterfactuals
        self.importances = importances
        self.count_diffs = count_diffs
        self.count_diffs_norm = count_diffs_norm

    def interpret_explanations(self, n_important_features: int = 3) -> str:
        """
        This method prints a report of the important features obtaiend.

        Args:
            n_important_features:
                The number of imporant features that will appear in the report.
                Defaults to 3.
        """

        importances_str = []
        for n in range(n_important_features):
            importance_str = [imp if isinstance(imp, str) else '{:.2f}'.format(imp) for imp in self.importances[n]]
            importances_str.append(importance_str)

        count_diffs_norm_str = []
        for n in range(n_important_features):
            count_diffs_i = list(self.count_diffs_norm.items())[n]
            count_diff_norm_str = '{} {:.2f}'.format(count_diffs_i[0], count_diffs_i[1])
            count_diffs_norm_str.append(count_diff_norm_str)

        interptretation = """The {} most important features and their importance values according to the first metric (amount features change) are: 
    {}.

According to the second metric (times features change), these importances are: 
    {}""".format(
            n_important_features, 
            ' AND '.join([' '.join(imp_str) for imp_str in importances_str]),
            ' AND '.join(count_diffs_norm_str)
        )
        print(interptretation)
        return interptretation
