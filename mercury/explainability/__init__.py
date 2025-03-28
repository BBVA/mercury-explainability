__version__ = '1.1.4'

from .explainers.counter_fact_basic import CounterFactualExplainerBasic
from .explainers.shuffle_importance import ShuffleImportanceExplainer
from .explainers.explainer import MercuryExplainer
from .explainers.partial_dependence import PartialDependenceExplainer
from .explanations.anchors import AnchorsWithImportanceExplanation
from .explanations.counter_factual import CounterfactualWithImportanceExplanation

from .explainers import ALEExplainer
from .explainers import AnchorsWithImportanceExplainer
from .explainers import CounterfactualExplainer, CounterfactualProtoExplainer

from .create_tutorials import create_tutorials
