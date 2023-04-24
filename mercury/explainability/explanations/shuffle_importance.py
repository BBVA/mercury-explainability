import numpy as np
import matplotlib.pyplot as plt


class FeatureImportanceExplanation():
    def __init__(self, data:dict, reverse:bool = False):
        """
        This class holds the data related to the importance a given
        feature has for a model.

        Args:
            data (dict):
                Contains the result of the PartialDependenceExplainer. It must be in the
                form of: ::
                    {
                        'feature_name': 1.0,
                        'feature_name2': 2.3, ...
                    }

            reverse (bool):
                Whether to reverse sort the features by increasing order (i.e. Worst
                performance (latest) = Smallest value). Default False (decreasing order).
        """
        self.data = data
        self._sorted_features = sorted(list(data.items()), key=lambda i: i[1],
                                       reverse=not reverse)

    def plot(self, ax: "matplotlib.axes.Axes" = None,  # noqa:F821
             figsize: tuple = (15, 15), limit_axis_x=False, **kwargs) -> "matplotlib.axes.Axes":  # noqa:F821
        """
        Plots a summary of the importances for each feature

        Args:
            figsize (tuple): Size of the plotted figure
            limit_axis_x (bool): Whether to adjust axis x to limit between the minimum and maximum feature values
        """
        ax = ax if ax else plt.gca()

        feature_names = [i[0] for i in self._sorted_features]
        feature_values = [i[1] for i in self._sorted_features]
        ax.barh(feature_names, feature_values)

        if limit_axis_x:
            ax.set_xlim(min(feature_values), max(feature_values))

        return ax

    def __getitem__(self, key:str)->float:
        """
        Gets the feature importance of the desired feature.

        Args:
            key (str): Name of the feature.
        """
        return self.data[key]

    def get_importances(self)->list:
        """ Returns a list of tuples (feature, importance) sorted by importances.
        """
        return self._sorted_features
