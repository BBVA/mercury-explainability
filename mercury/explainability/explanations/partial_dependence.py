import numpy as np
import matplotlib.pyplot as plt
import typing as TP

from math import ceil

class PartialDependenceExplanation():
    """
    This class holds the result of a Partial Dependence explanation and
    provides functionality for plotting those results via Partial Dependence Plots.

    Args:
        data (dict):
            Contains the result of the PartialDependenceExplainer. It must be in the
            form of: ::
                {
                    'feature_name': {'values': [...], 'preds': [...], 'lower_quantile': [...], 'upper_quantile': [...]},
                    'feature_name2': {'values': [...], 'preds': [...], 'lower_quantile': [...], 'upper_quantile': [...]},
                    ...
                }

    """
    def __init__(self, data):
        self.data = data

    def plot_single(self, var_name: str, ax=None, quantiles:TP.Union[bool, list] = False, filter_classes:list = None, **kwargs):
        """
        Plots the partial dependence of a single variable.

        Args:
            var_name (str):
                Name of the desired variable to plot.
            quantiles (bool or list[bool]):
                Whether to also plot the quantiles and a shaded area between them. Useful to check whether the predictions
                have high or low dispersion. If data doesn't contain the quantiles this parameter will be ignored.
            filter_clases (list):
                List of bool with the classes to plot. If None, all classes will be plotted. Ignored if the target variable
                is not categorical.
            ax (matplotlib.axes._subplots.AxesSubplot):
                Axes object on which the data will be plotted.
        """
        # If user pass a single bool and prediction data is a multinomial, we conver the
        # single boolean to a mask array to only plot the quantile range over the selected
        # classes.
        if len(self.data[var_name]['preds'].shape)>=2:
            if type(quantiles) == list and len(quantiles) != self.data[var_name]['preds'].shape[1]:
                raise ValueError("len(quantiles) must be equal to the number of classes.")
            if type(quantiles) == bool:
                quantiles = [quantiles for i in range(self.data[var_name]['preds'].shape[1])]
        elif type(quantiles) == list and len(self.data[var_name]['preds'].shape)==1:
            quantiles = quantiles[0]

        if filter_classes is not None:
            filter_classes = np.where(filter_classes)[0].tolist()
        else:
            filter_classes = np.arange(self.data[var_name]['preds'].shape[-1]).tolist()
            if len(self.data[var_name]['preds'].shape) < 2:
                filter_classes = None

        ax = ax if ax else plt.gca()

        ax.set_title(var_name)
        ax.set_xlabel(f"{var_name} value")
        ax.set_ylabel("Avg model prediction")

        vals = np.array(self.data[var_name]['values'])
        int_locations = np.arange(len(vals))

        non_numerical_values = False
        # Check if variable is categorical. If so, plot bars
        if self.data[var_name]['categorical'] and not type(vals[0]) == float:
            bar_width = .2
            class_nb = 0 if not filter_classes else len(filter_classes)

            if type(vals[0]) == float or type(vals[0]) == int:
                ax.set_xticks(self.data[var_name]['values'])
            else:
                non_numerical_values = True
                bar_offsets = np.linspace(-bar_width, bar_width, num=class_nb) / class_nb
                ax.set_xticks(int_locations)
                ax.set_xticklabels(self.data[var_name]['values'])

            if class_nb == 0:
                # If prediction is a single scalar
                if non_numerical_values:
                    ax.bar(int_locations, self.data[var_name]['preds'], width=bar_width, label='Prediction',**kwargs)
                else:
                    ax.bar(vals, self.data[var_name]['preds'], width=bar_width, label='Prediction', **kwargs)

                if quantiles:
                    ax.errorbar(
                        int_locations,
                        self.data[var_name]['preds'],
                        yerr=np.vstack([self.data[var_name]['lower_quantile'],
                                        self.data[var_name]['upper_quantile']]),
                        fmt='ko',
                        label='Quantiles',
                        **kwargs
                    )

            else:
                # If prediction is multiclass
                for i in range(class_nb):
                    if i in filter_classes:
                        if non_numerical_values:
                            ax.bar(int_locations + bar_offsets[i], self.data[var_name]['preds'][:,i],
                                    width=bar_width / class_nb, label=f'Class {i}',**kwargs)
                        else:
                            ax.bar(vals, self.data[var_name]['preds'][:,i], width=bar_width / class_nb, label=f'Class {i}', **kwargs)

                    if quantiles[i]:
                        ax.errorbar(
                            int_locations + bar_offsets[i],
                            self.data[var_name]['preds'][:, i],
                            yerr=np.vstack([self.data[var_name]['lower_quantile'][:,i],
                                            self.data[var_name]['upper_quantile'][:,i]]),
                            fmt='ko',
                            label=f'Quantiles {i}',
                            **kwargs
                        )

            if class_nb > 0:
                ax.legend()

        else:  # Variable is continuous

            # Check whether prediction data is multinomial
            if filter_classes:
                objs = ax.plot(vals, self.data[var_name]['preds'][:, filter_classes], **kwargs)
            else:
                objs = ax.plot(vals, self.data[var_name]['preds'], **kwargs)
            if len(self.data[var_name]['preds'].shape)>=2:
                labels = [f"Class: {i}" for i in range(self.data[var_name]['preds'].shape[1])]
                # Filter labels
                labels = [l for i, l in enumerate(labels) if i in filter_classes]
                # Show labels
                ax.legend(iter(objs), labels)
                for i in range(self.data[var_name]['preds'].shape[1]):
                    if quantiles[i] and len(self.data[var_name]['lower_quantile']) > 0:
                        # Plot quantiles and a shaded band between them

                        # We will need the color assigned to each one of the lines so the
                        # shaded area also has that color. Since filtering can be done, we
                        # extract the line index as the minimum between the current class
                        # index and the maximum amount of lines on the canvas.
                        obj_index = min(i, len(objs) - 1)

                        # Actually plot the shaded area
                        ax.plot(vals, self.data[var_name]['lower_quantile'][:,i], ls='--', color=objs[obj_index].get_color(),**kwargs)
                        ax.plot(vals, self.data[var_name]['upper_quantile'][:,i], ls='--', color=objs[obj_index].get_color(), **kwargs)
                        ax.fill_between(vals,
                                self.data[var_name]['lower_quantile'][:,i], self.data[var_name]['upper_quantile'][:,i], alpha=.05)
            else:  # If target is not multinomial
                if quantiles and len(self.data[var_name]['lower_quantile']) > 0:
                    # Plot quantiles and a shaded band between them
                    ax.plot(vals, self.data[var_name]['lower_quantile'], ls='--', color=objs[0].get_color(),**kwargs)
                    ax.plot(vals, self.data[var_name]['upper_quantile'], ls='--', color=objs[0].get_color(), **kwargs)
                    ax.fill_between(vals, self.data[var_name]['lower_quantile'], self.data[var_name]['upper_quantile'], alpha=.05)

    def plot(self, ncols:int = 1, figsize:tuple = (15,15), quantiles:TP.Union[bool, list] = False, filter_classes:list = None, **kwargs):
        """
        Plots a summary of all the partial dependences.

        Args:
            ncols (int):
                Number of columns of the summary. 1 as default.
            quantiles (bool or list):
                Whether to also plot the quantiles and a shaded area between them. Useful to check whether the predictions
                have high or low dispersion. If this is a list of booleans, quantiles
                will be plotted filtered by class (i.e. `quantiles[0]` = `class number 0`).
            filter_clases (list):
                List of bool with the classes to plot. If None, all classes will be plotted. Ignored if the target variable
                is not categorical.
            figsize (tuple):
                Size of the plotted figure
        """
        features = list(self.data.keys())

        fig, ax = plt.subplots(ceil(len(features) / ncols), ncols, figsize=figsize)

        for i, feat_name in enumerate(features):
            sbplt = ax[i] if ncols==1 or ncols==len(features) else ax[i // ncols, i % ncols]
            self.plot_single(feat_name, sbplt, quantiles=quantiles, filter_classes=filter_classes, **kwargs)

    def __getitem__(self, key:str):
        """
        Gets the dependence data of the desired feature.

        Args:
            key (str):
                Name of the feature.
        """
        return self.data[key]['values'], self.data[key]['preds']
