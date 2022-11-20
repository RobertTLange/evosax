from .control_gym import GymFitness
from .vision import VisionFitness
from .bbob import BBOBFitness
from .sequence import SequenceFitness

ProblemMapper = {
    "Gym": GymFitness,
    "Vision": VisionFitness,
    "BBOB": BBOBFitness,
    "Sequence": SequenceFitness,
}

__all__ = [
    "GymFitness",
    "VisionFitness",
    "BBOBFitness",
    "SequenceFitness",
    "ProblemMapper",
]
