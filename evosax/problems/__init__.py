from .control_brax import BraxFitness
from .control_gym import GymFitness
from .vision import VisionFitness
from .classic import ClassicFitness

ProblemMapper = {
    "Gym": GymFitness,
    "Brax": BraxFitness,
    "Vision": VisionFitness,
    "Classic": ClassicFitness,
}

__all__ = [
    "BraxFitness",
    "GymFitness",
    "VisionFitness",
    "ClassicFitness",
    "ProblemMapper",
]
