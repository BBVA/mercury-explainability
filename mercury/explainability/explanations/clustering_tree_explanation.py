
from typing import List
import numpy as np
try:
    from graphviz import Source
    graphviz_available = True
except Exception:
    graphviz_available = False


class ClusteringTreeExplanation():

    """
    Explanation for ClusteringTreeExplainer. Represents a Decision Tree for the explanation of a clustering
    algorithm.
    Using the plot method generates a visualization of the decision tree (requires graphviz package)

    Args:
        tree: the fitted decision tree
        feature_names: the feature names used in the decision tree

    """

    def __init__(
        self,
        tree: "Node",  # noqa: F821
        feature_names: List = None,
    ):
        self.tree = tree
        self.feature_names = feature_names

    def plot(self, filename: str = "tree_explanation", feature_names: List = None, scalers: dict = None):

        """
        Generates a graphviz.Source object representing the decision tree, which can be visualized in a notebook
        or saved in a file.

        Args:
            filename: filename to save if render() method is called over the returned object
            feature_names: the feature names to use. If not specified, the feature names specified in the constructor
                are used.
            scalers: dictionary of scalers. If passed, the tree will show the denormalized value in the split instead
                of the normalized value. The key is the feature name and the scaler must have the `inverse_transform`
                method

        Returns:
            (graphviz.Source): object representing the decision tree.
        """

        feature_names = self.feature_names if feature_names is None else feature_names
        scalers = {} if scalers is None else scalers

        if not graphviz_available:
            raise Exception("Required package is missing. Please install graphviz")

        if self.tree is not None:
            dot_str = ["digraph ClusteringTree {\n"]
            queue = [self.tree]
            nodes = []
            edges = []
            id = 0
            while len(queue) > 0:
                curr = queue.pop(0)
                if curr.is_leaf():
                    label = "%s\nsamples=\%d\nmistakes=\%d" % (str(self._get_node_split_value(curr)), curr.samples, curr.mistakes) # noqa
                else:
                    feature_name = curr.feature if feature_names is None else feature_names[curr.feature]
                    condition = "%s <= %.3f" % (feature_name, self._get_node_split_value(curr, feature_name, scalers))
                    label = "%s\nsamples=\%d" % (condition, curr.samples) # noqa
                    queue.append(curr.left)
                    queue.append(curr.right)
                    edges.append((id, id + len(queue) - 1))
                    edges.append((id, id + len(queue)))
                nodes.append({"id": id,
                              "label": label,
                              "node": curr})
                id += 1
            for node in nodes:
                dot_str.append("n_%d [label=\"%s\"];\n" % (node["id"], node["label"]))
            for edge in edges:
                dot_str.append("n_%d -> n_%d;\n" % (edge[0], edge[1]))
            dot_str.append("}")
            dot_str = "".join(dot_str)
            s = Source(dot_str, filename=filename + '.gv', format="png")
            return s

    def _get_node_split_value(self, node, feature_name=None, scalers=None):
        if (feature_name is not None) and (scalers is not None) and (feature_name in scalers):
            return scalers[feature_name].inverse_transform(np.array([node.value]).reshape(1, -1))[0][0]
        else:
            return node.value
