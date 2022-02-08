# Import additional utilities - Logging, visualization
from .es_logger import ESLog

# Import additional utilities for reshaping flat parameters into net dict
from .reshape_params import ParameterReshaper

# Import additional utilities for reshaping fitness
from .reshape_fitness import FitnessShaper

# Import Gradient Based Optimizer step functions
from .optimizer import (
    SGD_Optimizer,
    Adam_Optimizer,
    RMSProp_Optimizer,
    ClipUp_Optimizer,
)

GradientOptimizer = {
    "sgd": SGD_Optimizer,
    "adam": Adam_Optimizer,
    "rmsprop": RMSProp_Optimizer,
    "clipup": ClipUp_Optimizer,
}


__all__ = [
    "ESLog",
    "ParameterReshaper",
    "FitnessShaper",
    "GradientOptimizer",
    "SGD_Optimizer",
    "Adam_Optimizer",
    "RMSProp_Optimizer",
    "ClipUp_Optimizer",
]
