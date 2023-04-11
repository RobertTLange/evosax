from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from .simple_ga import single_mate
from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    sigma: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    cross_over_rate: float = 0.0
    sigma_init: float = 0.07
    sigma_ratio: float = 0.15
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class MR15_GA(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        elite_ratio: float = 0.0,
        sigma_ratio: float = 0.15,
        sigma_init: float = 0.1,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """1/5 MR Genetic Algorithm (Rechenberg, 1987)
        Reference: https://link.springer.com/chapter/10.1007/978-3-642-81283-5_8
        """

        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            n_devices=n_devices,
            **fitness_kwargs
        )
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
        self.strategy_name = "MR15_GA"

        # Set core kwargs es_params
        self.sigma_ratio = sigma_ratio  # no. mutation that have to improve
        self.sigma_init = sigma_init

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(
            sigma_init=self.sigma_init, sigma_ratio=self.sigma_ratio
        )

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the differential evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.elite_popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.elite_popsize) + jnp.finfo(jnp.float32).max,
            sigma=params.sigma_init,
            best_member=initialization.mean(axis=0),
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """
        `ask` for new proposed candidates to evaluate next.
        1. For each member of elite:
          - Sample two current elite members (a & b)
          - Cross over all dims of a with corresponding one from b
            if random number > co-rate
          - Additionally add noise on top of all elite parameters
        """
        rng, rng_eps, rng_idx_a, rng_idx_b = jax.random.split(rng, 4)
        rng_mate = jax.random.split(rng, self.popsize)
        epsilon = (
            jax.random.normal(rng_eps, (self.popsize, self.num_dims))
            * state.sigma
        )
        elite_ids = jnp.arange(self.elite_popsize)
        idx_a = jax.random.choice(rng_idx_a, elite_ids, (self.popsize,))
        idx_b = jax.random.choice(rng_idx_b, elite_ids, (self.popsize,))
        members_a = state.archive[idx_a]
        members_b = state.archive[idx_b]
        x = jax.vmap(single_mate, in_axes=(0, 0, 0, None))(
            rng_mate, members_a, members_b, params.cross_over_rate
        )
        x += epsilon
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """
        `tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        # Combine current elite and recent generation info
        fitness = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[0 : self.elite_popsize]
        fitness = fitness[idx]
        archive = solution[idx]
        # Update mutation sigma - double if more than 15% improved
        good_mutations_ratio = jnp.mean(fitness < state.best_fitness)
        increase_sigma = good_mutations_ratio > params.sigma_ratio
        sigma = jax.lax.select(
            increase_sigma, 2 * state.sigma, 0.5 * state.sigma
        )
        # Set mean to best member seen so far
        improved = fitness[0] < state.best_fitness
        best_mean = jax.lax.select(improved, archive[0], state.best_member)
        return state.replace(
            fitness=fitness, archive=archive, sigma=sigma, mean=best_mean
        )
