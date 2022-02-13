from .control_brax import BraxFitness
from .control_gym import GymFitness
from .supervised import SupervisedFitness
from .classic import ClassicFitness


__all__ = [
    "BraxFitness",
    "GymFitness",
    "SupervisedFitness",
    "ClassicFitness",
]
