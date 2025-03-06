"""Simulated Annealing (Rere et al., 2015).

[1] https://www.sciencedirect.com/science/article/pii/S1877050915035759
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct
import optax

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: Solution
    fitness: Fitness
    std: float
    temperature: float


@struct.dataclass
class Params(Params):
    temperature_init: float
    temperature_limit: float
    temperature_decay: float
    boltzmann_constant: float


class SimulatedAnnealing(DistributionBasedAlgorithm):
    """Simulated Annealing (SA)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize SA."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # std schedule
        self.std_schedule = std_schedule

    @property
    def _default_params(self) -> Params:
        return Params(
            temperature_init=1.0,
            temperature_limit=0.1,
            temperature_decay=0.999,
            boltzmann_constant=5.0,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            fitness=jnp.inf,
            std=self.std_schedule(0),
            temperature=params.temperature_init,
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
        z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + state.std * z
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        best_idx = jnp.argmin(fitness)
        best_member, best_fitness = population[best_idx], fitness[best_idx]

        delta = best_fitness - state.fitness
        metropolis = jnp.exp(-delta / (params.boltzmann_constant * state.temperature))
        acceptance = jax.random.uniform(key)

        # Replace if improved or random metropolis acceptance
        replace = jnp.logical_or(delta < 0, acceptance < metropolis)

        # Note: We replace by best member in generation
        mean = jnp.where(replace, best_member, state.mean)
        fitness = jnp.where(replace, best_fitness, state.fitness)

        # Update temperature
        temperature = jnp.clip(
            state.temperature * params.temperature_decay,
            min=params.temperature_limit,
        )

        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            fitness=fitness,
            temperature=temperature,
        )
