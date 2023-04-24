import signal


def run_until_timeout(timeout, fn, *args, **kwargs):
    """
    Timeout function to stop the execution in case it takes longer than the timeout argument.
    After timeout seconds it will raise an exception.

    Args:
        timeout: Number of seconds until the Exception is raised.
        fn: Function to execute.
        *args: args of the function fn.
        **kwargs: keyword args passed to fn.

    Example:
        >>> explanation = run_until_timeout(timeout, explainer.explain, data=data)

    """
    def signal_handler(signum, frame):
        raise Exception('Timeout: explanation took too long...')

    if timeout > 0:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(timeout)
    else:
        signal.alarm(0)
    return fn(*args, **kwargs)


from .counter_fact_basic import CounterFactualExplainerBasic
from .shuffle_importance import ShuffleImportanceExplainer
from .partial_dependence import PartialDependenceExplainer
from .clustering_tree_explainer import ClusteringTreeExplainer
from .explainer import MercuryExplainer

# Classes with alibi dependencies
try:
    from .ale import ALEExplainer
    from .anchors import AnchorsWithImportanceExplainer
    from .counter_fact_importance import CounterfactualExplainer, CounterfactualProtoExplainer
except ModuleNotFoundError:
    # if import fails, then import Dummy Class with the same name which raises Error if instantiated
    from ._dummy_alibi_explainers import (
        ALEExplainer,
        AnchorsWithImportanceExplainer,
        CounterfactualExplainer,
        CounterfactualProtoExplainer
    )
