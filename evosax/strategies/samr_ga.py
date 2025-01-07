from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
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
    sigma_init: float = 0.07
    sigma_meta: float = 2.0
    sigma_best_limit: float = 0.0001
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class SAMR_GA(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        elite_ratio: float = 0.0,
        sigma_init: float = 0.07,
        sigma_meta: float = 2.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Self-Adaptation Mutation Rate (SAMR) GA."""

        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            n_devices=n_devices,
            **fitness_kwargs
        )
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
        self.strategy_name = "SAMR_GA"

        # Set core kwargs es_params
        self.sigma_init = sigma_init
        self.sigma_meta = sigma_meta

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(sigma_init=self.sigma_init, sigma_meta=self.sigma_meta)

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
            sigma=jnp.zeros(self.elite_popsize) + params.sigma_init,
            best_member=initialization.mean(axis=0),
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        rng, rng_idx, rng_eps_x, rng_eps_s = jax.random.split(rng, 4)
        eps_x = jax.random.normal(rng_eps_x, (self.popsize, self.num_dims))
        eps_s = jax.random.uniform(
            rng_eps_s, (self.popsize,), minval=-1, maxval=1
        )
        idx = jax.random.choice(
            rng_idx, jnp.arange(self.elite_popsize), (self.popsize - 1,)
        )
        x = jnp.concatenate([state.archive[0][None, :], state.archive[idx]])
        sigma_0 = jnp.array(
            [jnp.maximum(params.sigma_best_limit, state.sigma[0])]
        )
        sigma = jnp.concatenate([sigma_0, state.sigma[idx]])
        sigma_gen = sigma * params.sigma_meta ** eps_s
        x += sigma_gen[:, None] * eps_x
        return x, state.replace(archive=x, sigma=sigma_gen)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        idx = jnp.argsort(fitness)[: self.elite_popsize]
        fitness = fitness[idx]
        archive = x[idx]
        sigma = state.sigma[idx]

        # Set mean to best member seen so far
        improved = fitness[0] < state.best_fitness
        best_mean = jax.lax.select(improved, archive[0], state.best_member)
        return state.replace(
            fitness=fitness, archive=archive, sigma=sigma, mean=best_mean
        )
