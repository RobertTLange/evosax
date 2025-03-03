"""Optimization problems module.

The problems are organized into several categories:
- Black-box optimization benchmark problems
- Reinforcement learning
- Computer vision

This module also contains neural network architectures and activation functions for use
with optimization problems.

Each problem implements a common interface with `sample` and `eval` methods to sample
solutions from the search space and evaluate their fitness, respectively.
"""

# BBOB
from .bbob.bbob import BBOBProblem
from .bbob.bbob_fns import bbob_fns
from .bbob.meta_bbob import MetaBBOBProblem

# Networks
from .networks import (
    CNN,
    MLP,
    categorical_output_fn,
    identity_output_fn,
    tanh_output_fn,
)

# RL
from .rl.brax import BraxProblem
from .rl.gymnax import GymnaxProblem

# Vision
from .vision.torchvision import TorchVisionProblem

bbob = [
    "BBOBProblem",
    "MetaBBOBProblem",
    "bbob_fns",
]

rl = [
    "GymnaxProblem",
    "BraxProblem",
]

vision = [
    "TorchVisionProblem",
]

networks = [
    "MLP",
    "CNN",
    "identity_output_fn",
    "categorical_output_fn",
    "tanh_output_fn",
]

__all__ = bbob + rl + vision + networks
