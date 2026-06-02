import matplotlib.pyplot as plt


class MonotonicityExplanation:
    """
    Class to represent the monotonicity explanation of a feature.

    Args:
        values (dict): Dictionary with monotonicity explanation values.
            Expected keys are:
            - feature (str): The feature name.
            - nmi_value (float): The Normalized Monotonicity Index value.
            - x_grid (list or np.array): The grid of feature values.
            - y_curve (list or np.array): The corresponding predicted values.
    """
    def __init__(self, values: dict):
        self.feature = values.get('feature')
        self.nmi_value = values.get('nmi_value')
        self.x_grid = values.get('x_grid')
        self.y_curve = values.get('y_curve')

    def __repr__(self):
        return f"MonotonicityExplanation(feature={self.feature}, nmi_value={self.nmi_value})"

    def plot(self, figsize=(8, 5), marker='o', title=None, xlabel=None, ylabel='Predicted Value', grid=True):
        plt.figure(figsize=figsize)
        plt.plot(self.x_grid, self.y_curve, marker=marker)
        plt.title(title if title else f'PDP Curve for {self.feature}\nNMI: {self.nmi_value:.3f}')
        plt.xlabel(xlabel if xlabel else self.feature)
        plt.ylabel(ylabel)
        if grid:
            plt.grid()
        plt.show()
