from .control_brax import BraxFitness
from .control_gym import GymFitness
from .vision import VisionFitness
from .classic import ClassicFitness
from .sequence import SequenceFitness

ProblemMapper = {
    "Gym": GymFitness,
    "Brax": BraxFitness,
    "Vision": VisionFitness,
    "Classic": ClassicFitness,
    "Sequence": SequenceFitness,
}

__all__ = [
    "BraxFitness",
    "GymFitness",
    "VisionFitness",
    "ClassicFitness",
    "SequenceFitness",
    "ProblemMapper",
]
