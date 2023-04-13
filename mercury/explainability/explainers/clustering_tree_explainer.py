
import typing as TP
import numpy as np
from collections import Counter
import sklearn
from sklearn.pipeline import Pipeline as SklearnPipeline
import pandas as pd

from ._tree_splitters import get_min_mistakes_cut, get_min_surrogate_cut
from .explainer import MercuryExplainer
from mercury.explainability.explanations.clustering_tree_explanation import ClusteringTreeExplanation


BASE_TREE = ['IMM', 'NONE']

LEAF_DATA_KEY_X_DATA = 'X_DATA_KEY'
LEAF_DATA_KEY_Y = 'Y_KEY'
LEAF_DATA_KEY_X_CENTER_DOT = 'X_CENTER_DOT'
LEAF_DATA_KEY_SPLITTER = 'SPLITTER_KEY'

class Node:
    def __init__(self):
        self.feature = None
        self.value = None
        self.samples = None
        self.mistakes = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    def set_condition(self, feature, value):
        self.feature = feature
        self.value = value


class Tree:

    def __init__(self, base_tree, k, max_leaves, all_centers, verbose, n_jobs):
        self.root = None
        self.base_tree = base_tree
        self.k = k
        self.max_leaves = max_leaves
        self.all_centers = all_centers
        self.verbose = verbose
        self.n_jobs = n_jobs
        self._leaves_data = {}

    def fit(self, X, y):
        if self.base_tree == "IMM":
            self.root = self._build_tree(X, y,
                                         np.ones(self.all_centers.shape[0], dtype=np.int32),
                                         np.ones(self.all_centers.shape[1], dtype=np.int32))
            leaves = self.k
        else:
            self.root = Node()
            self.root.value = 0
            leaves = 1

        if self.max_leaves > leaves:
            self.__gather_leaves_data__(self.root, X, y)
            all_centers_norm_sqr = (np.linalg.norm(self.all_centers, axis=1) ** 2).astype(np.float64, copy=False)
            self.__expand_tree__(leaves, all_centers_norm_sqr)
            self._leaves_data = {}

        self._feature_importance = np.zeros(X.shape[1])
        self.__fill_stats__(self.root, X, y)

    def _build_tree(self, X, y, valid_centers, valid_cols):
        """
        Build a tree.
        """
        if self.verbose >= 1:
            print('build node (samples=%d)' % X.shape[0])
        node = Node()
        if X.shape[0] == 0:
            node.value = 0
            return node
        elif valid_centers.sum() == 1:
            node.value = np.argmax(valid_centers)
            return node
        else:
            if np.unique(y).shape[0] == 1:
                node.value = y[0]
                return node
            else:

                # Verify data type is float64 prior to cython call
                X = X.astype(np.float64, copy=False)
                y = y.astype(np.int32, copy=False)
                self.all_centers = self.all_centers.astype(np.float64, copy=False)
                valid_centers = valid_centers.astype(np.int32, copy=False)
                valid_cols = valid_cols.astype(np.int32, copy=False)

                cut = get_min_mistakes_cut(X, y, self.all_centers, valid_centers, valid_cols, self.n_jobs)

                if cut is None:
                    node.value = np.argmax(valid_centers)
                else:
                    col = cut["col"]
                    threshold = cut["threshold"]
                    node.set_condition(col, threshold)

                    left_data_mask = X[:, col] <= threshold
                    matching_centers_mask = self.all_centers[:, col][y] <= threshold
                    mistakes_mask = left_data_mask != matching_centers_mask

                    left_valid_centers_mask = self.all_centers[valid_centers.astype(bool), col] <= threshold
                    left_valid_centers = np.zeros(valid_centers.shape, dtype=np.int32)
                    left_valid_centers[valid_centers.astype(bool)] = left_valid_centers_mask
                    right_valid_centers = np.zeros(valid_centers.shape, dtype=np.int32)
                    right_valid_centers[valid_centers.astype(bool)] = ~left_valid_centers_mask

                    node.left = self._build_tree(X[left_data_mask & ~mistakes_mask],
                                                 y[left_data_mask & ~mistakes_mask],
                                                 left_valid_centers,
                                                 valid_cols)
                    node.right = self._build_tree(X[~left_data_mask & ~mistakes_mask],
                                                  y[~left_data_mask & ~mistakes_mask],
                                                  right_valid_centers,
                                                  valid_cols)

                return node

    def predict(self, X: TP.Union[pd.DataFrame, np.array]):
        """
        Predict clusters for X.
            X: The input samples.
        Returns:
            The predicted clusters.
        """
        X = _convert_input(X)
        return self._predict_subtree(self.root, X)

    def _predict_subtree(self, node: Node, X: np.array):
        if node.is_leaf():
            return np.full(X.shape[0], node.value)
        else:
            ans = np.zeros(X.shape[0])
            left_mask = X[:, node.feature] <= node.value
            ans[left_mask] = self._predict_subtree(node.left, X[left_mask])
            ans[~left_mask] = self._predict_subtree(node.right, X[~left_mask])
            return ans

    def _size(self):
        """
        Return the number of nodes in the threshold tree.
        """
        return self.__size__(self.root)

    def __size__(self, node: Node):
        """
        Return the number of nodes in the subtree rooted by node.
        Args:
            node: root of a subtree.
        Returns
            the number of nodes in the subtree rooted by node.
        """
        if node is None:
            return 0
        else:
            sl = self.__size__(node.left)
            sr = self.__size__(node.right)
            return 1 + sl + sr

    def _max_depth(self):
        """
        Return the depth of the threshold tree.
        """
        return self.__max_depth__(self.root)

    def __max_depth__(self, node: Node):
        """
        Return the depth of the subtree rooted by node.

        Args:
            node: root of a subtree.

        Returns:
            The depth of the subtree rooted by node.
        """
        if node is None:
            return -1
        else:
            dl = self.__max_depth__(node.left)
            dr = self.__max_depth__(node.right)
            return 1 + max(dl, dr)

    def __expand_tree__(self, size, all_centers_norm_sqr):
        if size < self.max_leaves:
            if self.verbose >= 1:
                print('expand tree. size %d/%d' % (size, self.max_leaves))

            best_splitter = None
            leaf_to_split = None
            leaf_count = 1
            for leaf in self._leaves_data:
                if self.verbose >= 1:
                    print('-- expand leaf. %d/%d (samples=%d)' % (
                        leaf_count, len(self._leaves_data), self._leaves_data[leaf][LEAF_DATA_KEY_X_DATA].shape[0]))
                if LEAF_DATA_KEY_SPLITTER not in self._leaves_data[leaf]:
                    self._leaves_data[leaf][LEAF_DATA_KEY_SPLITTER] = self.__expand_leaf__(leaf, all_centers_norm_sqr)
                leaf_splitter = self._leaves_data[leaf][LEAF_DATA_KEY_SPLITTER]
                if leaf_splitter is not None:
                    if (best_splitter is None) or \
                        (best_splitter is not None and leaf_splitter["cost_gain"] < best_splitter["cost_gain"]):
                        best_splitter = leaf_splitter
                        leaf_to_split = leaf
                leaf_count += 1
            if best_splitter is not None:
                col = best_splitter["col"]
                threshold = best_splitter["threshold"]
                self.__split_leaf__(leaf_to_split,
                                    col,
                                    threshold,
                                    best_splitter["center_left"],
                                    best_splitter["center_right"])

                X = self._leaves_data[leaf_to_split][LEAF_DATA_KEY_X_DATA]
                y = self._leaves_data[leaf_to_split][LEAF_DATA_KEY_Y]
                X_center_dot = self._leaves_data[leaf_to_split][LEAF_DATA_KEY_X_CENTER_DOT]
                left_mask = X[:, col] <= threshold

                del self._leaves_data[leaf_to_split]

                self._leaves_data[leaf_to_split.left] = {LEAF_DATA_KEY_X_DATA: X[left_mask],
                                                         LEAF_DATA_KEY_Y: y[left_mask],
                                                         LEAF_DATA_KEY_X_CENTER_DOT: X_center_dot[left_mask]}
                self._leaves_data[leaf_to_split.right] = {LEAF_DATA_KEY_X_DATA: X[~left_mask],
                                                          LEAF_DATA_KEY_Y: y[~left_mask],
                                                          LEAF_DATA_KEY_X_CENTER_DOT: X_center_dot[~left_mask]}
                self.__expand_tree__(size + 1, all_centers_norm_sqr)

    def __gather_leaves_data__(self, node, X, y):
        if node.is_leaf():
            self._leaves_data[node] = {LEAF_DATA_KEY_X_DATA: X,
                                       LEAF_DATA_KEY_Y: y,
                                       LEAF_DATA_KEY_X_CENTER_DOT: np.dot(X, self.all_centers.T).astype(np.float64,
                                                                                                             copy=False)}
        else:
            left_mask = X[:, node.feature] <= node.value
            self.__gather_leaves_data__(node.left, X[left_mask], y[left_mask])
            self.__gather_leaves_data__(node.right, X[~left_mask], y[~left_mask])

    def __expand_leaf__(self, leaf, all_centers_norm_sqr):
        leaf_data = self._leaves_data[leaf]
        mistakes_counter = Counter([curr_y for curr_y in leaf_data[LEAF_DATA_KEY_Y] if curr_y != leaf.value])
        if len(mistakes_counter) == 0:
            return None

        # Verify data type is float64 prior to cython call
        X = leaf_data[LEAF_DATA_KEY_X_DATA].astype(np.float64, copy=False)
        X_center_dot = leaf_data[LEAF_DATA_KEY_X_CENTER_DOT].astype(np.float64, copy=False)
        all_centers_norm_sqr = all_centers_norm_sqr.astype(np.float64, copy=False)

        min_cut = get_min_surrogate_cut(X, X_center_dot, X_center_dot.sum(axis=0), all_centers_norm_sqr, self.n_jobs)

        if min_cut is not None:
            pre_split_cost = self.__get_leaf_pre_split_cost__(X_center_dot, all_centers_norm_sqr)
            splitter = {"col": min_cut["col"],
                        "threshold": min_cut["threshold"],
                        "cost_gain": min_cut["cost"] - pre_split_cost,
                        "center_left": min_cut["center_left"],
                        "center_right": min_cut["center_right"]}
            return splitter
        else:
            return None

    def __get_leaf_pre_split_cost__(self, X_center_dot, all_centers_norm_sqr):
        n = X_center_dot.shape[0]

        cost_per_center = (n * all_centers_norm_sqr) - 2 * X_center_dot.sum(axis=0)
        best_center = cost_per_center.argmin()
        return cost_per_center[best_center]

    def __split_leaf__(self, leaf, feature, value, left_cluster, right_cluster):
        leaf.feature = feature
        leaf.value = value

        leaf.left = Node()
        leaf.left.value = left_cluster

        leaf.right = Node()
        leaf.right.value = right_cluster

    def __fill_stats__(self, node, X, y):
        node.samples = X.shape[0]
        if not node.is_leaf():
            self._feature_importance[node.feature] += 1
            left_mask = X[:, node.feature] <= node.value
            self.__fill_stats__(node.left, X[left_mask], y[left_mask])
            self.__fill_stats__(node.right, X[~left_mask], y[~left_mask])
        else:
            node.mistakes = len([cluster for cluster in y if cluster != node.value])

    def feature_importance(self):
        return self._feature_importance


class ClusteringTreeExplainer(MercuryExplainer):
    """

    ClusteringTreeExplainer explains a clustering model using a DecisionTree.

    It is based on the method Iterative Mistake Minimization (IMM).
    The high-level idea is to find build a decision tree with the same number of leaves as the number of clusters. The tree is build
    by using the predicted clusters of a dataset using some previously fitted clustering model (like K-means) and fitting the
    decision tree using the clusters as labels. At each step, the a node with containing two or more of reference centres is split
    so the resulting split sends at least one reference centre to each side and moreover produces the fewest mistakes: that is,
    separates the minimum points from their corresponding centres.

    There is also de option to create a decision tree with a higher number of leaves than clusters. This is based on the ExKMC method,
    which is an extension of the IMM algorithm. The goal in this case is to achieve fewer mistakes in the resulting decision tree,
    with the trade-off that it will be less explainable. You can see more details of the methods in the referenced papers below

    In this implementation, the clustering solution can be created before using the ClusteringTreeExplainer. Otherwise,
    a k-means with default parameters is created before fitting the decision tree.

    References:
        **"Explainable k-Means and k-Medians Clustering"**: (http://proceedings.mlr.press/v119/moshkovitz20a/moshkovitz20a.pdf)
        **"ExKMC: Expanding Explainable k-Means Clustering"**: (https://arxiv.org/pdf/2006.02399.pdf)

    Args:
        clustering_model: The clustering model fitted. It supports sklearn and pyspark.
            When using sklearn, the model must be a sklearn `BaseEstimator` with a `predict` method and `cluster_centers_` attribute
            (like KMeans). Alternatively, you can provide a fitted pipeline where the last stage is the clustering algorithm.
            When using pyspark, the clustering_model must be a fitted Pyspark Estimator containing `clusterCenters()` method (like pyspark
            Kmeans). Alternatively, you can provide a fitted Pyspark `PipelineModel` where the last stage of the pipeline contains
            the clustering algorithm and contains the `clusterCenters()' method.
        k: number of clusters
        max_leaves: the maximum number of leaves. If max_leaves == k, then the method is the Iterative Mistake Minimization (IMM). If
            max_leaves > k, then the method to expand the tree further than k leaves is ExKMC. It cannot be max_leaves < k
        verbose: whether to show some messages during the process. If 0, it doesn't show any message. If >=1 show messages.
        base_tree: method to use to build the tree. If `'IMM'` it build a tree with k leaves using the IMM method and then it expands it
            using the ExKMC method if max_leaves > k. If `None` then it uses ExKMC method of splitting nodes from the root node. The default
            and recommended option is `'IMM'`
        n_jobs: number of jobs
        random_state: seed to use

    Example:
        ```python
        >>> from mercury.explainability.explainers.clustering_tree_explainer import ClusteringTreeExplainer
        >>> from sklearn.cluster import KMeans
        >>> k = 4
        >>> kmeans = KMeans(k, random_state=42)
        >>> kmeans.fit(df)
        >>> clustering_tree_explainer = ClusteringTreeExplainer(clustering_model=kmeans, k=k, max_leaves=k)
        >>> explanation = clustering_tree_explainer.explain(df)
        >>> plot_explanation = explanation.plot(filename="explanation")
        >>> plot_explanation
        ```
    """

    def __init__(
        self,
        clustering_model: TP.Union["sklearn.base.BaseEstimator", "pyspark.ml.base.Model"],  # noqa: F821
        max_leaves: int,
        verbose: int = 0,
        base_tree:str = 'IMM',
        n_jobs: int = None,
        random_state:int = None
    ):

        self.clustering_model = clustering_model
        self.all_centers = self._get_cluster_centers()
        self.k = len(self.all_centers)
        self.max_leaves = self.k if max_leaves is None else max_leaves
        if self.max_leaves < self.k:
            raise Exception('max_trees must be greater or equal to number of clusters [%d < %d]' % (self.max_leaves, self.k))
        self.verbose = verbose
        if base_tree not in BASE_TREE:
            raise Exception(base_tree + ' is not a supported base tree')
        self.base_tree = base_tree
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.random_state = random_state

        self.tree = None
        self._feature_importance = None

    def explain(
        self,
        X: TP.Union["pandas.DataFrame", "pyspark.sql.DataFrame"],  # noqa: F821
        subsample: float = None,
    ) -> ClusteringTreeExplanation:

        """
        Create explanation for clustering algorithm.

        Args:
            X: inputs of the data that we are clustering
            subsample: percentage of `X` to subsample (only when using pyspark)

        Returns:
            ClusteringTreeExplanation object, which contains plot() method to display the built decision tree
        """

        feature_names = X.columns

        if isinstance(X, pd.DataFrame):
            X, y = self._get_cluster_labels_pandas(X)
        else:
            X, y = self._get_cluster_labels_pyspark(X, subsample)

        self.tree = Tree(
            base_tree=self.base_tree, max_leaves=self.max_leaves, all_centers=self.all_centers,
            verbose=self.verbose, n_jobs=self.n_jobs, k=self.k
        )
        self.tree.fit(X, y)

        return ClusteringTreeExplanation(self.tree.root, feature_names=feature_names)

    def _get_cluster_labels_pandas(self, X):

        X = _convert_input(X)
        y = np.array(self.clustering_model.predict(X), dtype=np.int32)

        return X, y

    def _get_cluster_labels_pyspark(self, X, subsample):
        # import pyspark here to avoid importing when just using pandas
        import pyspark

        # Obtain predictions
        y = self.clustering_model.transform(X)

        # Subsample
        if subsample is not None:
            y = y.sample(fraction=subsample, seed=self.random_state)

        # Transform to pandas/np.array
        data = y.toPandas()
        X = data[X.columns]
        if (len(X.columns) == 1) and (isinstance(X[X.columns[0]].values[0], pyspark.ml.linalg.DenseVector)):
            X = _convert_dense_vectors_to_np_array(X)
        X = _convert_input(X)
        y = data["prediction"].values
        del data

        return X, y

    def _get_cluster_centers(self):
        if isinstance(self.clustering_model, sklearn.base.BaseEstimator):
            return self._get_cluster_centers_sklearn()
        else:
            return self._get_cluster_centers_pyspark()

    def _get_cluster_centers_sklearn(self):
        if isinstance(self.clustering_model, SklearnPipeline):
            return self.clustering_model.steps[-1][-1].cluster_centers_
        else:
            return self.clustering_model.cluster_centers_

    def _get_cluster_centers_pyspark(self):

        import pyspark

        if isinstance(self.clustering_model, pyspark.ml.pipeline.PipelineModel):
            all_centers = np.array(self.clustering_model.stages[-1].clusterCenters())
        else:
            all_centers = np.array(self.clustering_model.clusterCenters())

        return all_centers

    def score(self, X: TP.Union[pd.DataFrame, np.array]):
        """
        Return the k-means cost of X.
        The k-means cost is the sum of squared distances of each point to the mean of points associated with the cluster.
        Args:
            X: The input samples.
        Returns:
            k-means cost of X.
        """
        X = _convert_input(X)
        clusters = self.tree.predict(X)
        cost = 0
        for c in range(self.k):
            cluster_data = X[clusters == c, :]
            if cluster_data.shape[0] > 0:
                center = cluster_data.mean(axis=0)
                cost += np.linalg.norm(cluster_data - center) ** 2
        return cost

    def surrogate_score(self, X: TP.Union[pd.DataFrame, np.array]):
        """
        Return the k-means surrogate cost of X.
        The k-means surrogate cost is the sum of squared distances of each point to the closest center of the kmeans given (or trained)
        in the fit method. k-means surrogate cost > k-means cost, as k-means cost is computed with respect to the optimal centers.
        Args:
            X: The input samples.
        Returns:
            k-means surrogate cost of X.
        """
        X = _convert_input(X)
        clusters = self.tree.predict(X)
        cost = 0
        for c in range(self.k):
            cluster_data = X[clusters == c, :]
            if cluster_data.shape[0] > 0:
                center = self.all_centers[c]
                cost += np.linalg.norm(cluster_data - center) ** 2
        return cost


def _convert_input(data):
    if isinstance(data, list):
        data = np.array(data, dtype=np.float64)
    elif isinstance(data, np.ndarray):
        data = data.astype(np.float64, copy=False)
    elif isinstance(data, pd.DataFrame):
        data = data.values.astype(np.float64, copy=False)
    else:
        raise Exception(type(data) + ' is not supported type')
    return data

def _convert_dense_vectors_to_np_array(input_data):
    input_data = input_data[input_data.columns[0]].values
    n_samples = len(input_data)
    n_features = len(input_data[0])
    output_data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        output_data[i] = input_data[i].toArray()
    return output_data
