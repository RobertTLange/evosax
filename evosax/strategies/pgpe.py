import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from ..utils import GradientOptimizer, OptState, OptParams
from flax import struct


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: chex.Array
    opt_state: OptState
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    opt_params: OptParams
    sigma_init: float = 0.03
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    sigma_lrate: float = 0.2  # Learning rate for std
    sigma_max_change: float = 0.2  # Clip adaptive sigma to 20%
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class PGPE(Strategy):
    def __init__(
        self,
        num_dims: int,
        popsize: int,
        elite_ratio: float = 0.1,
        opt_name: str = "sgd",
    ):
        """PGPE (e.g. Sehnke et al., 2010)
        Reference: https://tinyurl.com/2p8bn956
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(num_dims, popsize)
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize / 2 * self.elite_ratio)

        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "PGPE"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(opt_params=self.optimizer.default_params)

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            sigma=jnp.ones(self.num_dims) * params.sigma_init,
            opt_state=self.optimizer.initialize(params.opt_params),
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        z_plus = jax.random.normal(
            rng,
            (int(self.popsize / 2), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.mean + z * state.sigma.reshape(1, self.num_dims)
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """Update both mean and dim.-wise isotropic Gaussian scale."""
        # Reconstruct noise from last mean/std estimates
        noise = (x - state.mean) / state.sigma
        noise_1 = noise[: int(self.popsize / 2)]
        fit_1 = fitness[: int(self.popsize / 2)]
        fit_2 = fitness[int(self.popsize / 2) :]
        elite_idx = jnp.minimum(fit_1, fit_2).argsort()[: self.elite_popsize]

        fitness_elite = jnp.concatenate([fit_1[elite_idx], fit_2[elite_idx]])
        fit_diff = fit_1[elite_idx] - fit_2[elite_idx]
        fit_diff_noise = jnp.dot(noise_1[elite_idx].T, fit_diff)

        theta_grad = 1.0 / self.elite_popsize * fit_diff_noise
        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        # Update sigma vector
        S = (
            noise_1 * noise_1
            - (state.sigma * state.sigma).reshape(1, self.num_dims)
        ) / state.sigma.reshape(1, self.num_dims)
        rS = (fit_1 + fit_2) / 2.0 - jnp.mean(fitness_elite)
        delta_sigma = (jnp.dot(rS, S)) / self.elite_popsize
        change_sigma = params.sigma_lrate * delta_sigma
        change_sigma = jnp.minimum(
            change_sigma, params.sigma_max_change * state.sigma
        )
        change_sigma = jnp.maximum(
            change_sigma, -params.sigma_max_change * state.sigma
        )

        # adjust sigma according to the adaptive sigma calculation
        # for stability, don't let sigma move more than 20% of orig value
        sigma = state.sigma - change_sigma
        sigma = sigma * params.sigma_decay
        sigma = jnp.maximum(sigma, params.sigma_limit)
        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state)
