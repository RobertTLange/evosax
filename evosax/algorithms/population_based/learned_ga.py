"""Learned Genetic Algorithm (Lange et al., 2023).

[1] https://arxiv.org/abs/2304.03995
Note: This is an independent reimplementation which does not use the same meta-trained
checkpoint used to generate the results in the paper. It has been independently
meta-trained and tested on a handful of Brax tasks. The results may therefore differ
from the ones shown in the paper.
"""

import pkgutil
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, PyTree, Solution

from ...learned_evolution.les_tools import (
    FitnessFeatures,
    load_pkl_object,
)
from ...learned_evolution.lga_tools import (
    MutationAttention,
    SamplingAttention,
    SelectionAttention,
    tanh_age,
)
from ..base import update_best_solution_and_fitness
from .base import Params, PopulationBasedAlgorithm, State, metrics_fn


@struct.dataclass
class State(State):
    population: jax.Array  # Parents: Solution vectors
    fitness: jax.Array  # Parents: Fitness scores
    std: jax.Array  # Parents: Mutation strengths
    age: jax.Array  # Parents: 'Age' counter
    std_C: jax.Array  # Children: Mutation strengths
    best_solution_shaped: Solution
    best_fitness_shaped: float


@struct.dataclass
class Params(Params):
    std_init: float
    crossover_rate: float
    params: PyTree


class LearnedGA(PopulationBasedAlgorithm):
    """Learned Genetic Algorithm (LGA)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        params: PyTree | None = None,
        params_path: str | None = None,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize LGA."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 1.0

        # LGA components
        self.fitness_features = FitnessFeatures(centered_rank=True, z_score=True)
        self.selection_layer = SelectionAttention(2, 16)
        self.sampling_layer = SamplingAttention(2, 16)
        self.mutation_layer = MutationAttention(2, 16)

        if params is not None:
            # Set params provided
            self.lga_params = params
        elif params_path is not None:
            # Load params from checkpoint
            self.lga_params = load_pkl_object(params_path)
        else:
            # Load default params
            if self.num_dims > 50:
                ckpt_fname = "2023_04_lga_v7.pkl"
            else:
                ckpt_fname = "2023_04_lga_v4.pkl"
            data = pkgutil.get_data(__name__, f"../ckpt/lga/{ckpt_fname}")
            self.lga_params = load_pkl_object(data, pkg_load=True)

    @property
    def _default_params(self) -> Params:
        return Params(
            std_init=1.0,
            crossover_rate=0.0,
            params=self.lga_params,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        return State(
            population=jnp.full((self.num_elites, self.num_dims), jnp.nan),
            fitness=jnp.full(self.num_elites, jnp.inf),
            std=params.std_init * jnp.ones((self.num_elites,)),
            age=jnp.zeros(self.num_elites),
            std_C=jnp.zeros((self.population_size,)),
            best_solution_shaped=jnp.full((self.num_dims,), jnp.nan),
            best_fitness_shaped=jnp.inf,
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        key_idx, key_epsilon = jax.random.split(key)

        # Sample candidates with replacement given distribution
        age_features = tanh_age(state.age, state.generation_counter + 1e-8)
        F_E = self.fitness_features.apply(
            state.population, state.fitness, state.best_fitness_shaped
        )
        F_E = jnp.concatenate([F_E, age_features[..., None]], axis=-1)
        p = self.sampling_layer.apply(params.params["sampling"], F_E)
        idx = jax.random.choice(key_idx, self.num_elites, (self.population_size,), p=p)

        # Select children with sampled indices
        X_C = state.population[idx]
        f_C = state.fitness[idx]
        std_C = state.std[idx]

        # Perform mutation rate adaptation of solutions
        F_C_tilde = self.fitness_features.apply(X_C, f_C, state.best_fitness_shaped)
        std_C = self.mutation_layer.apply(
            params.params["mutation"], std_C[..., None], F_C_tilde
        )

        # Perform Gaussian scaled mutation
        epsilon = jax.random.normal(key_epsilon, (self.population_size, self.num_dims))
        x = X_C + std_C * epsilon
        return x, state.replace(std_C=jnp.squeeze(std_C, axis=-1))

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        fitness_all = jnp.concatenate([fitness, state.fitness])
        solution_all = jnp.concatenate([population, state.population])
        fitness_features = self.fitness_features.apply(
            solution_all, fitness_all, state.best_fitness_shaped
        )

        # Perform selection - either learned or hard truncation based
        F_X = fitness_features[: self.population_size]
        F_E = fitness_features[self.population_size :]
        select = self.selection_layer.apply(params.params["selection"], key, F_X, F_E)
        keep_parent = jnp.sum(select, axis=-1) == 0

        # Update population
        population = select @ population + keep_parent[:, None] * state.population
        fitness = select @ fitness + keep_parent * state.fitness
        std = select @ state.std_C[:, None] + keep_parent[:, None] * state.std[:, None]
        age = jnp.where(keep_parent, state.age + 1, 0)

        # Update best solution and fitness shaped
        best_solution_shaped, best_fitness_shaped = update_best_solution_and_fitness(
            population, fitness, state.best_solution_shaped, state.best_fitness_shaped
        )

        return state.replace(
            population=population,
            fitness=fitness,
            std=jnp.squeeze(std, axis=-1),
            age=age,
            best_solution_shaped=best_solution_shaped,
            best_fitness_shaped=best_fitness_shaped,
        )
