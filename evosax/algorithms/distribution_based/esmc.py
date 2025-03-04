"""Evolution Strategy with Meta-loss Clipping (Merchant et al., 2021).

Reference: https://arxiv.org/abs/2107.09661
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
    z: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float


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
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize ESMC."""
        assert population_size % 2 == 1, "Population size must be odd"
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        # Optimizer
        self.optimizer = optimizer

    @property
    def _default_params(self) -> Params:
        return Params(std_init=1.0)

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            z=jnp.zeros((self.population_size // 2, self.num_dims)),
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
        z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
        z = jnp.concatenate(
            [jnp.zeros((1, self.num_dims)), z_plus, -z_plus]
        )  # TODO: different from original paper
        x = state.mean + state.std[None, ...] * z
        return x, state.replace(z=z_plus)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        fitness_baseline, fitness = fitness[0], fitness[1:]
        fitness_plus = fitness[: (self.population_size - 1) // 2]
        fitness_minus = fitness[(self.population_size - 1) // 2 :]

        # Compute grad
        delta = jnp.minimum(fitness_plus, fitness_baseline) - jnp.minimum(
            fitness_minus, fitness_baseline
        )
        grad = jnp.dot(state.z.T, delta) / int((self.population_size - 1) / 2)

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(mean=mean, opt_state=opt_state)
