"""Population-based algorithms module."""

from .de import DE
from .diffusion import DiffusionEvolution
from .gesmr_ga import GESMR_GA
from .lga import LGA
from .mr15_ga import MR15_GA
from .pso import PSO
from .samr_ga import SAMR_GA
from .simple_ga import SimpleGA

population_based_algorithms = {
    "DE": DE,
    "DiffusionEvolution": DiffusionEvolution,
    "GESMR_GA": GESMR_GA,
    "LGA": LGA,
    "MR15_GA": MR15_GA,
    "PSO": PSO,
    "SAMR_GA": SAMR_GA,
    "SimpleGA": SimpleGA,
}

__all__ = list(population_based_algorithms.keys())
