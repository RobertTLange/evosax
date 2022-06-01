# Import additional utilities - Logging, visualization
from .es_logger import ESLog

# Import additional utilities for reshaping flat parameters into net dict
from .reshape_params import ParameterReshaper

# Import additional utilities for reshaping fitness
from .reshape_fitness import FitnessShaper

# Import Gradient Based Optimizer step functions
from .optimizer import (
    SGD,
    Adam,
    RMSProp,
    ClipUp,
)

GradientOptimizer = {
    "sgd": SGD,
    "adam": Adam,
    "rmsprop": RMSProp,
    "clipup": ClipUp,
}


__all__ = [
    "ESLog",
    "ParameterReshaper",
    "FitnessShaper",
    "GradientOptimizer",
    "SGD",
    "Adam",
    "RMSProp",
    "ClipUp",
]
