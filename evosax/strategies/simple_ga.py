import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..core import exp_decay
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
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class SimpleGA(Strategy):
    def __init__(
        self,
        popsize: int,
        pholder_params: chex.ArrayTree | chex.Array | None = None,
        elite_ratio: float = 0.5,
        sigma_init: float = 0.1,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        **fitness_kwargs: bool | int | float,
    ):
        """Simple Genetic Algorithm (Such et al., 2017)
        Reference: https://arxiv.org/abs/1712.06567
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py
        """
        super().__init__(popsize, pholder_params, **fitness_kwargs)
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
        self.strategy_name = "SimpleGA"

        # Set core kwargs es_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
        )

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
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
    ) -> tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next.
        1. For each member of elite:
          - Sample two current elite members (a & b)
          - Cross over all dims of a with corresponding one from b
            if random number > co-rate
          - Additionally add noise on top of all elite parameters
        """
        rng, rng_eps, rng_idx_a, rng_idx_b = jax.random.split(rng, 4)
        rng_mate = jax.random.split(rng, self.popsize)
        epsilon = (
            jax.random.normal(rng_eps, (self.popsize, self.num_dims)) * state.sigma
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
        """`tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        # Combine current elite and recent generation info
        fitness = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[0 : self.elite_popsize]
        fitness = fitness[idx]
        archive = solution[idx]
        # Update mutation epsilon - multiplicative decay
        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        # Set mean to best member seen so far
        improved = fitness[0] < state.best_fitness
        best_mean = jax.lax.select(improved, archive[0], state.best_member)
        return state.replace(
            fitness=fitness, archive=archive, sigma=sigma, mean=best_mean
        )


def single_mate(
    rng: chex.PRNGKey, a: chex.Array, b: chex.Array, cross_over_rate: float
) -> chex.Array:
    """Only cross-over dims for x% of all dims."""
    idx = jax.random.uniform(rng, (a.shape[0],)) > cross_over_rate
    cross_over_candidate = a * (1 - idx) + b * idx
    return cross_over_candidate
