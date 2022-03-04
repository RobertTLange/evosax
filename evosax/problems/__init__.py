from .control_brax import BraxFitness
from .control_gym import GymFitness
from .supervised import SupervisedFitness
from .classic import ClassicFitness

ProblemMapper = {
    "Gym": GymFitness,
    "Brax": BraxFitness,
    "Supervised": SupervisedFitness,
    "Classic": ClassicFitness,
}

__all__ = [
    "BraxFitness",
    "GymFitness",
    "SupervisedFitness",
    "ClassicFitness",
    "ProblemMapper",
]
