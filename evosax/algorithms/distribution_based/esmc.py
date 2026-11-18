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

from .base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)


@struct.dataclass
class State(BaseState):
    mean: jax.Array
    std: jax.Array
    opt_state: optax.OptState


@struct.dataclass
class Params(BaseParams):
    pass


class ESMC(DistributionBasedAlgorithm):
    """Evolution Strategy with Meta-loss Clipping (ESMC).

    The first two population members are the unperturbed mean; the average of
    their fitness serves as a baseline. The remaining members are mirrored
    perturbation pairs `(z, -z)` drawn from an isotropic Gaussian around the mean.
    Each perturbed fitness is clipped from above at the baseline before the
    antithetic gradient estimate is formed, and the mean is updated with an Optax
    optimizer. The standard deviation follows a fixed schedule.

    Args:
        population_size: Number of candidate solutions sampled and evaluated per
            generation. Must be at least 4.
        solution: Example solution PyTree. Only its structure and leaf shapes are
            used: they set the number of search dimensions and how flat search
            vectors are unravelled into the PyTrees returned by `ask`.
        optimizer: Optax optimizer applied to the estimated gradient to update the
            mean.
        std_schedule: Function mapping the generation counter to the standard
            deviation of the search distribution, e.g. an Optax schedule.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied to
            the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`.
        metrics_fn: Function `(key, population, fitness, state, params) -> metrics`
            evaluated in `tell`; its output is returned alongside the new state.

    References:
        - [Merchant et al. (2021)](https://arxiv.org/abs/2107.09661)
    """

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
        updates, opt_state = self.optimizer.update(grad, state.opt_state, state.mean)
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            opt_state=opt_state,
        )
