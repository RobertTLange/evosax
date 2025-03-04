"""OpenAI Evolution Strategy (Salimans et al. 2017).

Reference: https://arxiv.org/abs/1703.03864
Inspired by: https://github.com/hardmaru/estool/blob/master/es.py
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    opt_state: optax.OptState


@struct.dataclass
class Params(Params):
    std_init: float


class Open_ES(DistributionBasedAlgorithm):
    """OpenAI Evolution Strategy (OpenAI-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        use_antithetic_sampling: bool = True,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize OpenAI-ES."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        # Optimizer
        self.optimizer = optimizer

        # Antithetic sampling
        self.use_antithetic_sampling = use_antithetic_sampling

    @property
    def _default_params(self) -> Params:
        return Params(std_init=1.0)

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
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
        if self.use_antithetic_sampling:
            z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
            z = jnp.concatenate([z_plus, -z_plus])
        else:
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
        # Compute grad
        grad = jnp.dot(fitness, (population - state.mean) / state.std) / (
            self.population_size * state.std
        )

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(mean=mean, opt_state=opt_state)
