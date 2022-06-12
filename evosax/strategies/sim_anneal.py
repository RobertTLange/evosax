import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from flax import struct


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    temp: float
    replace_rng: float
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    temp_init: float = 1.0
    temp_limit: float = 0.1
    temp_decay: float = 0.999
    boltzmann_const: float = 5.0
    sigma_init: float = 0.05
    sigma_limit: float = 0.001
    sigma_decay: float = 0.999
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class SimAnneal(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Simulated Annealing (Rasdi Rere et al., 2015)
        Reference: https://www.sciencedirect.com/science/article/pii/S1877050915035759
        """
        super().__init__(num_dims, popsize)
        self.strategy_name = "SimAnneal"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        rng_init, rng_rep = jax.random.split(rng)
        initialization = jax.random.uniform(
            rng_init,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init,
            temp=params.temp_init,
            replace_rng=jax.random.uniform(rng_rep, ()),
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        rng_noise, rng_rep = jax.random.split(rng)
        # Sampling of N(0, 1) noise
        z = jax.random.normal(
            rng_noise,
            (self.popsize, self.num_dims),
        )
        x = state.mean + state.sigma * z
        return x, state.replace(replace_rng=jax.random.uniform(rng_rep, ()))

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        best_in_gen = jnp.argmin(fitness)
        gen_fitness, gen_member = fitness[best_in_gen], x[best_in_gen]
        improve_diff = state.best_fitness - gen_fitness
        improved = improve_diff > 0

        # Calculate temperature replacement constant (replace by best in gen)
        metropolis = jnp.exp(
            improve_diff / (state.temp * params.boltzmann_const)
        )

        # Replace mean either if improvement or random metropolis acceptance
        rand_replace = jnp.logical_or(improved, state.replace_rng > metropolis)
        # Note: We replace by best member in generation (not completely random)
        mean = jax.lax.select(rand_replace, gen_member, state.mean)

        # Update permutation standard deviation
        sigma = jax.lax.select(
            state.sigma > params.sigma_limit,
            state.sigma * params.sigma_decay,
            state.sigma,
        )

        temp = jax.lax.select(
            state.temp > params.temp_limit,
            state.temp * params.temp_decay,
            state.temp,
        )
        return state.replace(mean=mean, sigma=sigma, temp=temp)
