from .bbob import BBOBProblem
from .gymnax import GymnaxProblem
from .sequence import SequenceProblem
from .vision import VisionProblem

ProblemMapper = {
    "Gymnax": GymnaxProblem,
    "Vision": VisionProblem,
    "BBOB": BBOBProblem,
    "Sequence": SequenceProblem,
}

__all__ = [
    "GymnaxProblem",
    "VisionProblem",
    "BBOBProblem",
    "SequenceProblem",
    "ProblemMapper",
]
