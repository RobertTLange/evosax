"""Group Elite Selection of Mutation Rates Genetic Algorithm (Kumar, 2022).

Reference: https://arxiv.org/abs/2204.04817
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Population, Solution
from .base import Params, PopulationBasedAlgorithm, State, metrics_fn


@struct.dataclass
class State(State):
    population: jax.Array
    fitness: Fitness
    std: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    std_min: float
    std_max: float
    std_meta: float


class GESMR_GA(PopulationBasedAlgorithm):
    """Group Elite Selection of Mutation Rates (GESMR)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        num_groups: int = 1,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize GESMR."""
        assert (population_size - 1) % num_groups == 0, (
            "Population size must be divisible by number of std groups"
        )
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        self.elite_ratio = 0.5
        self.std_ratio = 0.5

        # Number of groups
        self.num_groups = num_groups

    @property
    def num_std_elites(self):
        """Get the number of std elites."""
        return max(1, int(self.std_ratio * self.num_groups))

    @property
    def members_per_group(self):
        """Get the number of members per group."""
        return (self.population_size - 1) // self.num_groups

    @property
    def _default_params(self) -> Params:
        return Params(
            std_init=1.0,
            std_min=0.0,
            std_max=10.0,
            std_meta=2.0,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full(self.population_size, jnp.inf),
            std=params.std_init * jnp.ones((self.num_groups,)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        key_idx, key_eps = jax.random.split(key)

        # Sort population by fitness
        idx = jnp.argsort(state.fitness)
        population = state.population[idx]
        fitness = state.fitness[idx]

        # Selection
        idx = jax.random.choice(key_idx, self.num_elites, (self.population_size - 1,))
        population = jnp.concatenate([population[:1], population[idx]])  # Eq. (3)
        fitness = jnp.concatenate([fitness[:1], fitness[idx]])
        state = state.replace(population=population, fitness=fitness)

        # Mutation
        eps = jax.random.normal(key_eps, (self.population_size, self.num_dims))
        std_repeated = jnp.repeat(state.std, self.members_per_group)
        std = jnp.concatenate([0.0 * state.std[:1], std_repeated])
        population += std[:, None] * eps  # Eq. (4)

        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        key_idx, key_eps = jax.random.split(key)

        # Compute delta
        segment_ids = jnp.repeat(jnp.arange(self.num_groups), self.members_per_group)
        delta = jax.ops.segment_min(
            fitness[1:] - state.fitness[1:],
            segment_ids,
            num_segments=self.num_groups,
            indices_are_sorted=True,
        )  # Eq. (5)

        # Sort std by delta
        idx = jnp.argsort(delta)
        std = state.std[idx]

        # Selection
        idx = jax.random.choice(key_idx, self.num_std_elites, (self.num_groups - 1,))
        std = jnp.concatenate([std[:1], std[idx]])  # Eq. (6)

        # Mutation
        eps = jax.random.uniform(key_eps, (self.num_groups,), minval=-1, maxval=1)
        std = std * params.std_meta**eps  # Eq. (7)
        std = jnp.clip(std, min=params.std_min, max=params.std_max)

        return state.replace(
            population=population,
            fitness=fitness,
            std=std,
        )
