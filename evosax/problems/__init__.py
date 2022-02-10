from .control_brax import BraxFitness
from .control_gym import GymnaxFitness
from .supervised import SupervisedFitness
from .classic import ClassicFitness


__all__ = [
    "BraxFitness",
    "GymnaxFitness",
    "SupervisedFitness",
    "ClassicFitness",
]
