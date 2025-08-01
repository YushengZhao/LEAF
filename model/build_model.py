from .graph import GraphBranch
from .hypergraph import HypergraphBranch


def get_model_class(name):
    try:
        return eval(name)
    except NameError:
        raise ValueError('Unknown model name')
