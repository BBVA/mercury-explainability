from typing import Union, Callable
from .explainer import MercuryExplainer
from ..explanations.shuffle_importance import FeatureImportanceExplanation


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class ShuffleImportanceExplainer(MercuryExplainer):
    """
    This explainer estimates the feature importance of each predictor for a
    given black-box model. The used strategy consists on random shuffling one
    variable at a time and, on each step, checking how much a particular
    metric worses. The features which make the model to perform the worst are
    the most important ones.

    Args:
        eval_fn (Callable):
            Custom evaluation function. It will recieve a DataFrame with features and
            a Numpy array with the target outputs for each instance. It must
            implement an inference process and return a metric to score the model
            performance on the given data. This metric must be real numbered.
            If we use a metric which higher values means better metric (like accuracy)
            and we use the parameter `normalize=True` (default option), then it is
            recommended to return the negative of that metric in `eval_fn` to make
            the results more intuitive.
            In the case of Pyspark explanations, the function will only recieve
            a PySpark in the first argument already containing the target column,
            whereas the second argument will be None.
        normalize (bool):
            Whether to scale the feature importances between 0 and 1. If True, then
            it shows the relative importance of the features.
            If False, then the feature importances will be the value of the metric
            returned in `eval_fn` when shuffling the features.
            Default value is True

    Example:
        ```python
        # "Plain python" example
        >>> features = pd.read_csv(PATH_DATA)
        >>> targets = features['target']  # Targets
        >>> features = features.loc[:, FEATURE_NAMES] # DataFrame with only features
        >>> def my_inference_function(features, targets):
        ...     predictions = model.predict(features)
        ...     return mean_squared_error(targets, predictions)
        >>> explainer = ShuffleImportanceExplainer(my_inference_function)
        >>> explanation = explainer.explain(features, targets)
        >>> explanation.plot()

        # Explain a pyspark model (or pipeline)
        >>> features = sess.createDataFrame(pandas_dataframe)
        >>> target_colname = "target"  # Column name with the ground truth labels
        >>> def my_inference_function(features, targets):
        ...     model_inp = vectorAssembler.transform(features)
        ...     model_out = my_pyspark_transformer.transform(model_inp)
        ...     return my_evaluator.evaluate(model_out)
        >>> explainer = ShuffleImportanceExplainer(my_inference_function)
        >>> explanation = explainer.explain(features, target_colname)
        >>> explanation.plot()
        ```
    """
    def __init__(self,
                 eval_fn: Callable[[Union["pd.DataFrame", "pyspark.sql.DataFrame"], Union["np.ndarray", str]], float],  # noqa: F821
                 normalize: bool = True
        ):
        self.eval_fn = eval_fn
        self.normalize = normalize

    def explain(self,
                predictors: Union["pd.DataFrame", "pyspark.sql.DataFrame"],  # noqa: F821
                target: Union["np.ndarray", str]
        ) -> FeatureImportanceExplanation:
        """
        Explains the model given the data.

        Args:
            predictors (Union[pandas.DataFrame, pyspark.sql.DataFrame]):
                DataFrame with the features the model needs and that will be explained.
                In the case of PySpark, this dataframe must  also contain a column
                with the target.
            target (Union[numpy.ndarray, str]):
                The ground-truth target for each one of the instances. In the case of
                Pyspark, this should be the name of the column in the DataFrame which
                holds the target.

        Raises:
            ValueError: if type(predictors) == pyspark.sql.DataFrame && type(target) != str
            ValueError: if type(predictors) == pyspark.sql.DataFrame && target not in predictors.columns

        Returns:
            FeatureImportanceExplanation with the performances of the model
        """

        implementation = self.__impl_base
        feature_names = []
        # Cheap way of check if type(predictors) == pyspark.sql.DataFrame (without importing pyspark).
        if hasattr(type(predictors), 'toPandas'):
            if type(target) != str:
                raise ValueError("""If predictors is a Spark DataFrame, target should be the name \
                                 of the tareget column (a str)""")
            implementation = self.__impl_pyspark
            feature_names = list(filter(lambda x: x!=target, predictors.columns))
            if len(feature_names) == len(predictors.columns):
                raise ValueError(f"""`target` must be the name of the target column in the DataFrame. \
                                 Value passed: {target}""")
        else:
            feature_names = list(predictors.columns)
            if type(target) == str:
                feature_names = list(filter(lambda x: x!=target, feature_names))

        metrics = {}
        for col in feature_names:
            metrics[col] = implementation(predictors, target, col)
        if self.normalize:
            metrics = self._normalize_importances(metrics)
        return FeatureImportanceExplanation(metrics)

    def __impl_base(self, predictors, target, column):
        temp = predictors.copy()
        temp[column] = np.array(temp[column].sample(frac=1))
        return self.eval_fn(temp, target)

    def __impl_pyspark(self, predictors, target, column):
        from pyspark.sql.functions import rand, row_number, monotonically_increasing_id
        from pyspark.sql.window import Window

        # Shuffle column values creating a temporal rand column and ordering by it.
        # Then, we merge the ordered DF with the old one removing the original and
        # temporal columns
        shuffled = predictors.select(column).withColumn('rand', rand()).orderBy('rand')
        # In order to merge back we also need to include a row number.
        shuffled=shuffled.withColumn('shuff_id_idx', row_number().over(Window.orderBy(monotonically_increasing_id())))
        temp = predictors.withColumn('shuff_id_idx', row_number().over(Window.orderBy(monotonically_increasing_id())))

        temp = temp.drop(column).join(shuffled, on='shuff_id_idx').drop('shuff_id_idx').drop('rand')

        return self.eval_fn(temp, None)

    def _normalize_importances(self, metrics):
        df_metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["importance"])
        df_metrics["importance"] = MinMaxScaler().fit_transform(df_metrics["importance"].values.reshape(-1,1))
        return df_metrics["importance"].to_dict()
