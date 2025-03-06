"""Population-based algorithms module."""

from .differential_evolution import DifferentialEvolution
from .diffusion_evolution import DiffusionEvolution
from .gesmr_ga import GESMR_GA
from .learned_ga import LearnedGA
from .mr15_ga import MR15_GA
from .pso import PSO
from .samr_ga import SAMR_GA
from .simple_ga import SimpleGA

population_based_algorithms = {
    "DifferentialEvolution": DifferentialEvolution,
    "DiffusionEvolution": DiffusionEvolution,
    "GESMR_GA": GESMR_GA,
    "LGA": LearnedGA,
    "MR15_GA": MR15_GA,
    "PSO": PSO,
    "SAMR_GA": SAMR_GA,
    "SimpleGA": SimpleGA,
}

__all__ = list(population_based_algorithms.keys())
