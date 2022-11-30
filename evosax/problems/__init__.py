from .control_gym import GymnaxFitness
from .vision import VisionFitness
from .bbob import BBOBFitness
from .sequence import SequenceFitness

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
