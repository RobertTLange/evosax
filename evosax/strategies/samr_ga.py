import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy


@struct.dataclass
class State:
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    sigma: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class Params:
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
        population_size: int,
        solution: chex.ArrayTree | chex.Array | None = None,
        elite_ratio: float = 0.0,
        sigma_init: float = 0.07,
        sigma_meta: float = 2.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Self-Adaptation Mutation Rate (SAMR) GA."""
        super().__init__(population_size, solution, **fitness_kwargs)
        self.elite_ratio = elite_ratio
        self.elite_population_size = max(
            1, int(self.population_size * self.elite_ratio)
        )
        self.strategy_name = "SAMR_GA"

        # Set core kwargs params
        self.sigma_init = sigma_init
        self.sigma_meta = sigma_meta

    @property
    def params_strategy(self) -> Params:
        """Return default parameters of evolution strategy."""
        return Params(sigma_init=self.sigma_init, sigma_meta=self.sigma_meta)

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """`init` the differential evolution strategy."""
        initialization = jax.random.uniform(
            key,
            (self.elite_population_size, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = State(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.elite_population_size) + jnp.finfo(jnp.float32).max,
            sigma=jnp.zeros(self.elite_population_size) + params.sigma_init,
            best_member=initialization.mean(axis=0),
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[chex.Array, State]:
        """`ask` for new proposed candidates to evaluate next."""
        key_idx, key_eps_x, key_eps_s = jax.random.split(key, 3)

        idx = jax.random.choice(
            key_idx, jnp.arange(self.elite_population_size), (self.population_size - 1,)
        )
        eps_x = jax.random.normal(key_eps_x, (self.population_size, self.num_dims))
        eps_s = jax.random.uniform(
            key_eps_s, (self.population_size,), minval=-1, maxval=1
        )

        sigma_0 = jnp.array([jnp.maximum(params.sigma_best_limit, state.sigma[0])])
        sigma = jnp.concatenate([sigma_0, state.sigma[idx]])
        sigma_gen = sigma * params.sigma_meta**eps_s

        x = jnp.concatenate([state.archive[0][None, :], state.archive[idx]])
        x += sigma_gen[:, None] * eps_x
        return x, state.replace(archive=x, sigma=sigma_gen)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: State,
        params: Params,
    ) -> State:
        """`tell` update to ES state."""
        idx = jnp.argsort(fitness)[: self.elite_population_size]
        fitness = fitness[idx]
        archive = x[idx]
        sigma = state.sigma[idx]

        # Set mean to best member seen so far
        improved = fitness[0] < state.best_fitness
        best_mean = jax.lax.select(improved, archive[0], state.best_member)
        return state.replace(
            fitness=fitness, archive=archive, sigma=sigma, mean=best_mean
        )
