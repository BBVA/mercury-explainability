import typing as TP
import pandas as pd
import numpy as np


class SparkWrapper:
    """
    This class is an adaptor which allows Spark models to also be
    explained by Mercury. In order to explain your model you should wrap it
    with this.

    Args:
        transformer: Trained PySpark model (transformer), or anything implementing a transform method
        feature_names: Name of the features the PySpark model uses.
        spark_session: Current spark session in use.
        model_inp_name: Name of the input column for the model (output name of the VectorAssembler)
        model_out_name: Output column name of the PySpark model. Default one is "probability".
        vector_assembler: If None, a default one will be created in order to transform the features to a Vector used by the PySpark model.
        probability_threshold: If >=0, the output of the model will be normalized taking into account the threshold.

    Example:
        ```python
        # model is a spark transformer (including a pipeline) already trained we want to explain
        >>> model_w = SparkWrapper(
        ...      model,
        ...      dataset.feature_names,
        ...      spark_session=spark,
        ...      model_inp_name="features",
        ...      model_out_name="probability",
        ... )
        # model_w is a model ready to be explained with Mercury explainers.
        ```
    """

    def __init__(self,
                 transformer: 'pyspark.ml.Transformer',  # noqa: F821
                 feature_names: list,
                 spark_session: 'pyspark.sql.SparkSession' = None,  # noqa: F821
                 model_inp_name: str = "features",
                 model_out_name: str = "probability",
                 vector_assembler: 'pyspark.sql.VectorAssembler' = None,  # noqa: F821
                 probability_threshold: float = -1
                 ):

        import pyspark

        self.model = transformer
        self.spark_session = spark_session
        self.feature_names = feature_names
        self.model_out_name = model_out_name
        self.vector_assembler = vector_assembler
        self.probability_threshold = probability_threshold

        if self.vector_assembler is None:
            self.vector_assembler = pyspark.ml.feature.VectorAssembler(
                inputCols=feature_names, outputCol=model_inp_name
            )

    @staticmethod
    def _transform_threshold(
        x: np.ndarray,
        threshold: float
    ) -> np.ndarray:

        if x[1] < threshold:
            x[1] = 0.5 * (x[1] / threshold)
        else:
            x[1] = 0.5 + 0.5 * ((x[1] - threshold) / (1 - threshold))
        x[0] = 1 - x[1]
        return x

    def __call__(self, data: TP.Union['pd.DataFrame', np.ndarray] = None):
        if data.shape[1] != len(self.feature_names):
            raise ValueError("The input does not have the same number of features as the specified in self.feature_names.")

        x_to_be_predicted = data
        if type(data) is np.ndarray:
            data = pd.DataFrame(data, columns=self.feature_names)
        sp_df_x = self.spark_session.createDataFrame(data)
        x_to_be_predicted = self.vector_assembler.transform(
            sp_df_x,
        ) if self.vector_assembler else sp_df_x

        pred_out = np.stack(self.model.transform(x_to_be_predicted)
                            .select(self.model_out_name).toPandas()[self.model_out_name].apply(
            lambda x: np.array(x.toArray())
        ).values)

        if self.probability_threshold >= 0:
            pred_out = np.apply_along_axis(
                lambda x: self._transform_threshold(x, self.probability_threshold),
                1,
                pred_out
            )

        return pred_out