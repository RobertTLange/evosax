"""Evolution Strategy with Meta-loss Clipping (Merchant et al., 2021).

[1] https://arxiv.org/abs/2107.09661
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    opt_state: optax.OptState


@struct.dataclass
class Params(Params):
    pass


class ESMC(DistributionBasedAlgorithm):
    """Evolution Strategy with Meta-loss Clipping (ESMC)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.adam(
            learning_rate=optax.exponential_decay(
                init_value=1e-2,
                transition_steps=10,
                decay_rate=0.98,
            )
        ),
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize ESMC."""
        assert population_size >= 4, "Population size must be >= 4"
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # Optimizer
        self.optimizer = optimizer

        # std schedule
        self.std_schedule = std_schedule

    @property
    def _default_params(self) -> Params:
        return Params()

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=self.std_schedule(0),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
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
        # Antithetic sampling
        z_plus = jax.random.normal(key, (self.population_size // 2 - 1, self.num_dims))
        z = jnp.concatenate([jnp.zeros((2, self.num_dims)), z_plus, -z_plus])
        x = state.mean + state.std * z
        return x, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        z_plus = (
            population[2 : self.population_size // 2 + 1] - state.mean
        ) / state.std

        fitness_baseline, fitness = jnp.mean(fitness[:2], axis=0), fitness[2:]
        fitness_plus = fitness[: self.population_size // 2 - 1]
        fitness_minus = fitness[self.population_size // 2 - 1 :]

        # Compute grad
        delta = jnp.minimum(fitness_plus, fitness_baseline) - jnp.minimum(
            fitness_minus, fitness_baseline
        )
        grad = jnp.dot(delta, z_plus) / int((self.population_size - 1) / 2)

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(mean=mean, std=self.std_schedule(state.generation_counter), opt_state=opt_state)
