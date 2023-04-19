import typing as TP
import numpy as np
import pandas as pd
import signal

from .explainer import MercuryExplainer
from mercury.explainability.explainers import run_until_timeout
from alibi.explainers import AnchorTabular
from mercury.explainability.explanations.anchors import AnchorsWithImportanceExplanation
from alibi.api.interfaces import Explanation


class AnchorsWithImportanceExplainer(AnchorTabular, MercuryExplainer):
    """
    Extending Alibi's AnchorsTabular Implementation, this module allows for the
    computation of feature importance by means of calculating several anchors.
    Initialize the anchor tabular explainer.

    Args:
        predict_fn: Model prediction function
        train_data: Pandas Dataframe with the features
        disc_perc: List or tuple with percentiles (int) used for discretization.
        categorical_names: Dictionary where keys are feature columns and values are the categories for the feature

    Raises:
        AttributeError: if categorical_names is not a dict
        AttributeError: if train_data is not a pd.DataFrame

    Example:
        ```python
        >>> explain_data = pd.read_csv('./test/explain_data.csv')
        >>> model = MyModel() # (Trained) model prediction function (has be callable)
        >>> explainer = AnchorsWithImportanceExplainer(model, explain_data)
        >>> explanation = explainer.explain(explain_data.head(10).values) # For the first 10 samples
        >>> explanation.interpret_explanations(n_important_features=2)
        # We can also get the feature importances for the first 10 samples.
        >>> anchorsExtendedExplainer.get_feature_importance(explain_data=explain_data.head(10))
        ```
    """

    def __init__(
        self,
        predict_fn: TP.Callable,
        train_data: pd.DataFrame,
        categorical_names: TP.Dict[str, TP.List] = {},
        disc_perc: TP.Tuple[TP.Union[int, float], ...] = (25, 50, 75),
        *args, **kwargs
    ) -> None:
        if not isinstance(categorical_names, dict):
            raise AttributeError("""
                The attribute categorical_names should be a dictionary
                where the keys are the categorical feature names and the
                values are the categories for each categorical feature.
            """)

        if not isinstance(train_data, pd.DataFrame):
            raise AttributeError("""
                train_data should be a pandas DataFrame.
            """)

        super().__init__(predict_fn, list(train_data.columns), categorical_names)
        self.categorical_names = categorical_names

        super().fit(
            train_data=train_data.values,
            disc_perc=disc_perc,
            *args, **kwargs
        )

    def explain(self,
                X: np.ndarray,
                threshold: float = 0.95,
                delta: float = 0.1,
                tau: float = 0.15,
                batch_size: int = 100,
                coverage_samples: int = 10000,
                beam_size: int = 1,
                stop_on_first: bool = False,
                max_anchor_size: TP.Optional[int] = None,
                min_samples_start: int = 100,
                n_covered_ex: int = 10,
                binary_cache_size: int = 10000,
                cache_margin: int = 1000,
                verbose: bool = False,
                verbose_every: int = 1,
                **kwargs: TP.Any) -> Explanation:
        """
        Explain prediction made by classifier on instance `X`.

        Args:
            X: Instance to be explained.
            threshold: Minimum precision threshold.
            delta: Used to compute `beta`.
            tau: Margin between lower confidence bound and minimum precision or upper bound.
            batch_size: Batch size used for sampling.
            coverage_samples: Number of samples used to estimate coverage from during result search.
            beam_size: The number of anchors extended at each step of new anchors construction.
            stop_on_first: If ``True``, the beam search algorithm will return the
                            first anchor that has satisfies the probability constraint.
            max_anchor_size: Maximum number of features in result.
            min_samples_start: Min number of initial samples.
            n_covered_ex: How many examples where anchors apply to store for each anchor sampled during search
                            (both examples where prediction on samples agrees/disagrees with `desired_label` are stored).
            binary_cache_size: The result search pre-allocates `binary_cache_size` batches for storing the binary arrays
                                returned during sampling.
            cache_margin: When only ``max(cache_margin, batch_size)`` positions in the binary cache remain empty, a new cache
                            of the same size is pre-allocated to continue buffering samples.
            verbose: Display updates during the anchor search iterations.
            verbose_every: Frequency of displayed iterations during anchor search process.

        Returns:
            explanation
                `Explanation` object containing the result explaining the instance with additional metadata as attributes.
                See usage at `AnchorTabular examples`_ for details.
                .. _AnchorTabular examples:
                    https://docs.seldon.io/projects/alibi/en/latest/methods/Anchors.html
        """
        exp = super().explain(
            X=X,
            threshold=threshold,
            delta=delta,
            tau=tau,
            batch_size=batch_size,
            coverage_samples=coverage_samples,
            beam_size=beam_size,
            stop_on_first=stop_on_first,
            max_anchor_size=max_anchor_size,
            min_samples_start=min_samples_start,
            n_covered_ex=n_covered_ex,
            binary_cache_size=binary_cache_size,
            cache_margin=cache_margin,
            verbose=verbose,
            verbose_every=verbose_every
        )

        # This attribute makes pickle serialization crash, so we delete it.
        if hasattr(self, "mab"):
            delattr(self, "mab")

        return exp

    def get_feature_importance(
        self,
        explain_data: pd.DataFrame,
        threshold: float = 0.95,
        print_every: int = 0,
        print_explanations: bool = False,
        n_important_features: int = 3,
        tau: float = 0.15,
        timeout: int = 0) -> AnchorsWithImportanceExplanation:
        """
        Args:
            explain_data:
                Pandas dataframe containing all the instances for which to find an anchor and therefore
                obtain feature importances.
            threshold: To be used in and passed down to the anchor explainer as defined on Alibi's documentation.
                Controls the minimum precision desired when looking for anchors.
                Defaults to 0.95.
            print_every:
                Logging information.
                Defaults to 0 - No logging
            print_explanations:
                Boolean that determines whether to print the explanations at the end of the method or not.
                Defaults to False.
            n_important_features:
                Number of top features that will be printed.
                Defaults to 3.
            tau:
                To be used in and passed down to the anchos explainer as defined on Alibi's documentation.
                Used within the multi-armed bandit part of the optimisation problem.
                Defaults to 0.15
            timeout:
                Maximum time to be spent looking for an Anchor in seconds. A value of 0 means that no timeout
                is set.
                Defaults to 0.

        Returns:
            A list containing all the explanations.
        """

        explanations = []
        if print_every > 0:
            print('Looking for a total of {} explanations'.format(
                len(explain_data))
            )
        for explain_datum_idx, explain_datum in explain_data.iterrows():
            try:
                explanation = run_until_timeout(timeout,
                                  self.explain,
                                  explain_datum.values,
                                  threshold=threshold,
                                  tau=tau)
                explanations.append(explanation)
            except Exception:
                if print_every > 0:
                    print('No anchor found for observation {}'
                                                .format(explain_datum_idx))
                explanations.append('No explanation')

            # Unset timeout
            signal.alarm(0)

            if print_every > 0:
                if len(explanations) % print_every == 0:
                    print(
                        ("""A total of {} observations have been processed """ +
                            """for explaining""").format(len(explanations)))
                    print("{} anchors have already been found".format(
                        sum([1 for explan in explanations
                            if not isinstance(explan, str)])
                    ))
        # Here we have a list with all the anchors explanations that we've been able to find.
        anchorsExtendedExplanation = AnchorsWithImportanceExplanation(
            explain_data=explain_data,
            explanations=explanations,
            categorical=self.categorical_names
        )
        if print_explanations:
            anchorsExtendedExplanation.interpret_explanations(
                n_important_features=n_important_features
            )
        return anchorsExtendedExplanation

    def translate(self, explanation: Explanation) -> str:
        """
        Translates an explanation into simple words

        Args:
            explanation: Alibi explanation object

        """
        coverage = explanation['data']['coverage']
        if type(explanation['data']['precision']) is np.ndarray:
            precision = explanation['data']['precision'][0]
        else:
            precision = explanation['data']['precision']

        if coverage * precision < 0.1:
            quality = "POOR"
        elif 0.1 <= coverage * precision < 0.4:
            quality = "GOOD"
        else:
            quality = "GREAT"

        return "[{} explanation] This anchor explains a {}% of all records of its class with {}% confidence.".format(
            quality,
            round(100 * coverage, 2),
            round(100 * precision, 2)
        )
