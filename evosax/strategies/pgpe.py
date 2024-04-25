from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..strategy import Strategy
from ..core import GradientOptimizer, OptState, OptParams, exp_decay


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
    sigma_init: float = 0.1
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    sigma_lrate: float = 0.1  # Learning rate for std
    sigma_max_change: float = 0.2  # Clip adaptive sigma to 20%
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class PGPE(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        elite_ratio: float = 1.0,
        opt_name: str = "adam",
        lrate_init: float = 0.15,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_init: float = 0.1,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """PGPE (e.g. Sehnke et al., 2010)
        Reference: https://tinyurl.com/2p8bn956
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(
            popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs
        )
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize / 2 * self.elite_ratio))

        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "PGPE"

        # Set core kwargs es_params (lrate/sigma schedules)
        self.lrate_init = lrate_init
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        opt_params = self.optimizer.default_params.replace(
            lrate_init=self.lrate_init,
            lrate_decay=self.lrate_decay,
            lrate_limit=self.lrate_limit,
        )
        return EvoParams(
            opt_params=opt_params,
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
        )

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
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
        z = jnp.hstack([z_plus, -1.0 * z_plus]).reshape(-1, self.num_dims)
        x = state.mean + state.sigma * z
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
        scaled_noise = x - state.mean
        noise_1 = scaled_noise[::2]
        fit_1 = fitness[::2]
        fit_2 = fitness[1::2]
        elite_idx = jnp.minimum(fit_1, fit_2).argsort()[: self.elite_popsize]
        fitness_elite = jnp.concatenate([fit_1[elite_idx], fit_2[elite_idx]])
        fit_diff = fit_1[elite_idx] - fit_2[elite_idx]
        fit_diff_noise = noise_1[elite_idx] * fit_diff[:, None]

        theta_grad = (0.5 * fit_diff_noise).mean(axis=0)
        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        baseline = jnp.mean(fitness_elite)
        all_avg_scores = jnp.stack([fit_1[elite_idx], fit_2[elite_idx]]).sum(axis=0) / 2
        # Update sigma vector
        delta_sigma = (
            (jnp.expand_dims(all_avg_scores, axis=1) - baseline)
            * (noise_1[elite_idx] ** 2 - jnp.expand_dims(state.sigma**2, axis=0))
            / state.sigma
        ).mean(axis=0)

        allowed_delta = jnp.abs(state.sigma) * params.sigma_max_change
        min_allowed = state.sigma - allowed_delta
        max_allowed = state.sigma + allowed_delta

        # adjust sigma according to the adaptive sigma calculation
        # for stability, don't let sigma move more than 20% of orig value
        sigma = jnp.clip(
            state.sigma + params.sigma_lrate * delta_sigma,
            min_allowed,
            max_allowed,
        )
        sigma = exp_decay(sigma, params.sigma_decay, params.sigma_limit)
        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state)
