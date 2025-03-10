"""Evolutionary algorithms module.

This module provides a collection of evolutionary algorithms implemented in JAX,
organized into two main categories:

1. Distribution-based algorithms: These algorithms maintain a probability distribution
    over the search space and sample from it to generate new candidate solutions.
    Examples include CMA-ES, NES variants, and simple hill-climbing methods.

2. Population-based algorithms: These algorithms maintain a population of candidate
   solutions and apply various operators (selection, crossover, mutation) to evolve
   the population. Examples include Genetic Algorithms, Differential Evolution,
   Particle Swarm Optimization, and Diffusion Evolution.
"""

# ruff: noqa: F401

# Distribution-based algorithms
from .distribution_based import distribution_based_algorithms
from .distribution_based.ars import ARS
from .distribution_based.asebo import ASEBO
from .distribution_based.cma_es import CMA_ES
from .distribution_based.cr_fm_nes import CR_FM_NES
from .distribution_based.discovered_es import DiscoveredES
from .distribution_based.esmc import ESMC
from .distribution_based.evotf_es import EvoTF_ES
from .distribution_based.gradientless_descent import GradientlessDescent
from .distribution_based.guided_es import GuidedES
from .distribution_based.hill_climbing import HillClimbing
from .distribution_based.iamalgam_full import iAMaLGaM_Full
from .distribution_based.iamalgam_univariate import iAMaLGaM_Univariate
from .distribution_based.learned_es import LearnedES
from .distribution_based.lm_ma_es import LM_MA_ES
from .distribution_based.ma_es import MA_ES
from .distribution_based.noise_reuse_es import NoiseReuseES
from .distribution_based.open_es import Open_ES
from .distribution_based.persistent_es import PersistentES
from .distribution_based.pgpe import PGPE
from .distribution_based.random_search import RandomSearch
from .distribution_based.rm_es import Rm_ES
from .distribution_based.sep_cma_es import Sep_CMA_ES
from .distribution_based.simple_es import SimpleES
from .distribution_based.simulated_annealing import SimulatedAnnealing
from .distribution_based.snes import SNES
from .distribution_based.sv.sv_cma_es import SV_CMA_ES
from .distribution_based.sv.sv_open_es import SV_Open_ES
from .distribution_based.xnes import xNES

# Population-based algorithms
from .population_based import population_based_algorithms
from .population_based.differential_evolution import DifferentialEvolution
from .population_based.diffusion_evolution import DiffusionEvolution
from .population_based.gesmr_ga import GESMR_GA
from .population_based.learned_ga import LearnedGA
from .population_based.mr15_ga import MR15_GA
from .population_based.pso import PSO
from .population_based.samr_ga import SAMR_GA
from .population_based.simple_ga import SimpleGA

# Combine algorithms from both categories
algorithms = distribution_based_algorithms | population_based_algorithms

__all__ = list(algorithms.keys())
