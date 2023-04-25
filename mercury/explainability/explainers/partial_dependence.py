import time
import pandas as pd
import numpy as np
import typing as TP

from .explainer import MercuryExplainer
from mercury.explainability.explanations.partial_dependence import PartialDependenceExplanation


class PartialDependenceExplainer(MercuryExplainer):
    """
    This explainer will calculate the partial dependences for a ML model.
    Also contains a distributed (pyspark) implementation which
    allows PySpark transformers/pipelines to be explained via PD.

    Args:
        predict_fn (Callable):
            Inference function. This function will take a DataFrame (Pandas
            or PySpark) and must return the output of your estimator (usually a NumPy array
            for a 'plain' python model or a DataFrame in case of PySpark).
        output_col (str):
            Name of the output column name  of the PySpark transformer model.
            This will only be used in case the passed explain data is of type
            pyspark.DataFrame
        max_categorical_thresh (int):
            If a column contains less unique values than this threshold, it will be
            auto-considered categorical.
        quantiles (float or tuple):
            Calculate the quantiles of the model predictions. If type==float,
            (quantiles, 1-quantiles) range will be calculated. If type==tuple, range
            will be `quantiles`. If None, the quantile calculation will be disabled
            (useful for saving time).
        resolution (int):
            Number of different values to test for each non-categorical variable. Lowering this
            value will increase speed but reduce resolution in the plots.
        verbose (bool):
            Print progress status. Default is False.

    Example:
        ```python
        # "Plain python" example
        >>> features = dataset.loc[:, FEATURE_NAMES] # DataFrame with only features
        # You can create a custom inference function.
        >>> def my_inference_fn(feats):
        ...     return my_fitted_model.predict(feats)
        # You can also pass anything as long as it is callable and receives the predictors (e.g. my_fitted_model.predict / predict_proba).
        >>> explainer = PartialDependenceExplainer(my_fitted_model.predict)
        >>> explanation = explainer.explain(features)
        # Plot a summary of partial dependences for all the features.
        >>> explanation.plot()
        # Plot the partial dependence of a single variable.
        >>> fig, ax = plt.subplots()
        >>> explanation.plot_single('FEATURE_1', ax=ax)

        # Pyspark Example
        # Dataset with ONLY feature columns. (Assumed an already trained model)
        >>> features = dataset.drop('target')
        # Inference function
        >>> def my_pred_fn(data):
        ...     temp_df = assembler.transform(data)
        ...     return my_transformer.transform(temp_df)
        # We can tell the explainer not to calculate  partial dependences for certain features. This will save time.
        >>> features_to_ignore = ['FEATURE_4', 'FEATURE_88']
        # We pass our custom inference function and also, tell the explainer which column will hold the transformer's output (necessary only when explaining pyspark models).
        >>> explainer = PartialDependenceExplainer(my_pred_fn, output_col='probability')
        # Explain the model ignoring features.
        >>> explanation = explainer.explain(features, ignore_feats=features_to_ignore)
        ```
    """
    def __init__(self,
            predict_fn: TP.Callable[[TP.Union["pandas.DataFrame", "pyspark.sql.DataFrame"]],  # noqa: F821
                                 TP.Union[np.ndarray, "pyspark.sql.DataFrame"]],  # noqa: F821
            output_col:str = 'prediction',
            max_categorical_thresh:int = 5,
            quantiles:TP.Union[float ,tuple] = 0.05,
            resolution: int = 50,
            verbose:bool = False):
        self.predict_fn = predict_fn
        self.output_col = output_col
        # If the number of different values of a given feature is less
        # than this, we will treat it as categorical
        self._max_categorical_values = max_categorical_thresh
        self.verbose = verbose
        self.resolution = resolution

        if type(quantiles) == float:
            self.compute_quantiles = (quantiles, 1 - quantiles)
        else:
            self.compute_quantiles = quantiles

    def explain(self,
            features: TP.Union["pandas.DataFrame", "pyspark.sql.DataFrame"],  # noqa: F821
            ignore_feats: TP.List[str] = None,
            categoricals: TP.List[str] = None)->PartialDependenceExplanation:
        """
        This method will compute the partial dependences for a ML model.
        This explainer also contains a distributed (pyspark) implementation which
        allows PySpark transformers/pipelines to be explained via PD.

        Args:
            features (pandas.DataFrame or pyspark.sql.DataFrame):
                DataFrame with only the features needed by the model.
                This dataframe should ONLY contain the features consumed by the model and,
                in the PySpark case, the vector-assembled column should be generated inside
                of the predict_fn.
            ignore_feats (List):
                Feature names which won't be explained
            categoricals (List):
                of feature names that will be forced to be taken as
                categoricals. If it's empty, the explainer will guess what columns are
                categorical

        Returns:
            PartialDependenceExplanation containing the explanation results.
        """
        ignore_feats = ignore_feats if ignore_feats else []
        categoricals = categoricals if categoricals else []

        partial_dep_impl = self.__base_impl
        categoricals = set(categoricals)
        if type(features) != pd.DataFrame:
            partial_dep_impl = self.__pyspark_impl

        feat_names = [f for f in list(features.columns) if f not in set(ignore_feats)]

        dependences = partial_dep_impl(features, feat_names, categoricals)

        return PartialDependenceExplanation(dependences)

    def __base_impl(self, features:pd.DataFrame, feat_names:list, categoricals:set)->dict:
        """
        Contains the logic for the base implementation, i.e. the  one
        which uses np.ndarrays/pd.DataFrames and undistributed models.

        Args:
            features (pd.DataFrame): pandas.DataFrame with ONLY the predictors.
            feat_names (list[str]):
                List containing the names of the features that will be explained
            categoricals (set[str]):
                Set with the feature names that will be forced to be categorical.

        Returns:
            Dictionary with the partial dependences of each selected feature
            and whether that feature is categorical or not.
        """
        data = {}
        for colname in feat_names:
            start = time.time()
            data[colname] = {}
            uniques = features[colname].unique()
            nb_uniques = len(uniques)

            if features.loc[:,colname].dtype == np.dtype('O') or colname in categoricals:
                is_categorical = True
            else:
                is_categorical = False

            if not is_categorical and type(features.loc[:,colname].iloc[0].item()) == float and nb_uniques > self._max_categorical_values:
                grid = np.linspace(features[colname].min(), features[colname].max())
            elif not is_categorical and nb_uniques > self._max_categorical_values and type(features.loc[:,colname].iloc[0].item()) == int:
                step_size = max(1, np.abs((features[colname].max() - features[colname].min()) // self.resolution))
                grid = np.arange(features[colname].min(), features[colname].max(), step=step_size)
            else:
                is_categorical = True
                grid = uniques

            if self.verbose:
                type_msg = 'categorical' if is_categorical else 'continuous'
                print(f"Starting calculation for feature {colname} - Type: {type_msg}")
                if is_categorical:
                    print(f"\tValues to test: {grid}")
                else:
                    print(f"\tValue range to test: [{grid[0]}, {grid[-1]}] (only on {self.resolution} items)")

            data[colname]['values'] = grid
            pdep_means, pdep_lquant, pdep_uquant = self.__partial_dep_base(colname, features, grid)
            data[colname]['preds'] = pdep_means
            data[colname]['lower_quantile'] = pdep_lquant
            data[colname]['upper_quantile'] = pdep_uquant
            data[colname]['categorical'] = is_categorical

            elapsed = time.time() - start
            if self.verbose:
                print(f"Partial dependence for feature {colname} calculated. Took {elapsed:.2f} seconds")

        return data

    def __pyspark_impl(self, features: "pyspark.DataFrame", feat_names: list, categoricals: set) -> dict:  # noqa: F821
        """ Contains the logic for the pyspark implementation, i.e. the  one
        which uses pyspark.DataFrames and undistributed models.

        Args:
            features: pyspark.DataFrame with ONLY the predictors.
            feat_names: list[str]
                List containing the names of the features that will be explained
            categoricals: set[str]
                Set with the feature names that will be forced to be categorical.

        Returns:
            Dictionary with the partial dependences of each selected feature
            and whether that feature is categorical or not.
        """
        from pyspark.sql.functions import col
        from pyspark.sql import functions as sqlf

        from pyspark.sql.types import (
            FloatType,
            DoubleType,
            DecimalType,
            LongType,
            IntegerType,
            ShortType,
            ByteType
        )

        # Pyspark names of float and integer datatypes
        float_pyspark_types = (FloatType, DoubleType, DecimalType)
        int_pyspark_types = (FloatType, DoubleType, LongType, IntegerType, ShortType, ByteType)

        data = {}

        # Calculate unique values for disjoint set of columns from categoricals
        cols_to_check = [n for n in feat_names if n not in categoricals]
        # Calculate all statistics at once to save collects
        num_distinct_values = features.agg(
            *(sqlf.countDistinct(col(c)).alias(c) for c in cols_to_check)
        ).collect()[0]
        minimums = features.select([sqlf.min(c).alias(c) for c in features.columns]).collect()[0]
        maximums = features.select([sqlf.max(c).alias(c) for c in features.columns]).collect()[0]

        for colname in feat_names:
            start = time.time()

            data[colname] = {}
            # Determine whether this column is discrete, real or categorical
            is_categorical = False
            if isinstance(features.schema[colname].dataType, float_pyspark_types) and colname not in categoricals:
                grid = np.linspace(minimums[colname], maximums[colname])
            elif isinstance(features.schema[colname].dataType, int_pyspark_types) \
                    and num_distinct_values[colname] > self._max_categorical_values:
                step_size = max(
                    1, np.abs((maximums[colname] - minimums[colname]) // self.resolution)
                )
                grid = np.arange(minimums[colname], maximums[colname], step=step_size)
            else:
                grid = features.select(colname).distinct().rdd.flatMap(lambda x: x).collect()
                is_categorical=True

            if self.verbose:
                type_msg = 'categorical' if is_categorical else 'continuous'
                print(f"Starting calculation for feature {colname} - Type: {type_msg}")
                if is_categorical:
                    print(f"\tValues to test: {grid}")
                else:
                    print(f"\tValue range to test: [{grid[0]}, {grid[-1]}] (only on {self.resolution} items)")

            data[colname]['values'] = grid
            pdep_means, pdep_lquant, pdep_uquant = self.__partial_dep_pyspark(colname, features, grid, is_categorical)
            data[colname]['preds'] = pdep_means
            data[colname]['lower_quantile'] = pdep_lquant
            data[colname]['upper_quantile'] = pdep_uquant
            data[colname]['categorical'] = is_categorical

            elapsed = time.time() - start
            if self.verbose:
                print(f"Partial dependence for feature {colname} calculated. Took {elapsed:.2f} seconds")

        return data

    def __partial_dep_base(self,
            feat_name:str,
            data: "pandas.DataFrame",  # noqa: F821
            grid: TP.Union[list, "np.array"])->"np.ndarray":
        """ Logic for computing the partial dependence of one feature for the base
        implementation (undistributed models)
        """
        preds = []
        lower_quantiles = []
        upper_quantiles = []
        data = data.copy()
        for val in grid:
            data[feat_name] = val
            model_preds = self.predict_fn(data)
            preds.append(np.mean(model_preds, axis=0))
            if self.compute_quantiles:
                lower_quantiles.append(np.quantile(model_preds, q=self.compute_quantiles[0],axis=0))
                upper_quantiles.append(np.quantile(model_preds, q=self.compute_quantiles[1],axis=0))
        return np.array(preds), np.array(lower_quantiles), np.array(upper_quantiles)

    def __partial_dep_pyspark(self,
            feat_name:str,
            data: TP.Union["pandas.DataFrame", "pyspark.sql.DataFrame"],  # noqa: F821
            grid: TP.Union[list, "np.array"],
            is_categorical:bool = False)->"np.ndarray":
        """ Helper method for the PySpark implementation (distributed models).
        Computes the partial dependences of one feature given its type (categorical or not)
        """
        from pyspark.ml.stat import Summarizer
        from pyspark.sql.functions import col
        from pyspark.sql import functions as sqlf

        mean_preds = []
        lower_quantiles = []
        upper_quantiles = []

        for v in grid:
            if type(v) == str:
                temp_df = data.withColumn(feat_name, sqlf.lit(v))
            else:
                temp_df = data.withColumn(feat_name, sqlf.lit(float(v)))
            temp_df = self.predict_fn(temp_df)

            if temp_df.schema[self.output_col].simpleString().split(':')[-1] == 'vector':
                # Aggregate all the probability distributions
                mean_preds.append(temp_df.agg(Summarizer.mean(col(self.output_col))).collect()[0][0].values.tolist())

                if self.compute_quantiles:
                    # Tricky to calculate quantiles as Spark doesnt support this for Vector
                    def _to_cols(row):
                        return tuple(row["probability"].toArray().tolist())

                    pred_cols = temp_df.rdd.map(_to_cols).toDF()
                    quants = np.array(
                        pred_cols.approxQuantile(pred_cols.columns, self.compute_quantiles, 0.15)
                    )
                    lower_quantiles.append(quants[:, 0])
                    upper_quantiles.append(quants[:, 1])
            else:
                # Aggregate all the prediction probabilities/targets
                mean_preds.append(temp_df.agg({self.output_col: "mean"}).collect()[0][0])
                if self.compute_quantiles:
                    quants =temp_df.approxQuantile(self.output_col, self.compute_quantiles, 0.15)
                    lower_quantiles.append(quants[0])
                    upper_quantiles.append(quants[1])

        return np.array(mean_preds), np.array(lower_quantiles), np.array(upper_quantiles)
