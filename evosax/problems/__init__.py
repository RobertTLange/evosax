from .bbob import BBOBFitness
from .control_gym import GymnaxFitness
from .sequence import SequenceFitness
from .vision import VisionFitness

ProblemMapper = {
    "Gymnax": GymnaxFitness,
    "Vision": VisionFitness,
    "BBOB": BBOBFitness,
    "Sequence": SequenceFitness,
}

__all__ = [
    "GymnaxFitness",
    "VisionFitness",
    "BBOBFitness",
    "SequenceFitness",
    "ProblemMapper",
]
