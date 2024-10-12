from typing import Any, Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..core import GradientOptimizer, OptState, OptParams, exp_decay
from .distributed import DistributedStrategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: chex.Array
    opt_state: OptState
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    clip_min: float
    clip_max: float
    opt_params: OptParams
    sigma_init: float = 0.1
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    sigma_lrate: float = 0.1  # Learning rate for std
    sigma_max_change: float = 0.2  # Clip adaptive sigma to 20%
    init_min: float = 0.0
    init_max: float = 0.0


@struct.dataclass
class EvoUpdate:
    grad_mean: chex.Array
    delta_sigma: chex.Array


class PGPE(DistributedStrategy):

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
        n_devices: int = 1,
        param_dtype: Any = jnp.float32,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """PGPE (e.g. Sehnke et al., 2010)
        Reference: https://tinyurl.com/2p8bn956
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            mean_decay,
            n_devices,
            param_dtype,
            **fitness_kwargs
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
            clip_min=jnp.finfo(self.param_dtype).min,
            clip_max=jnp.finfo(self.param_dtype).max,
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

    def get_update_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoUpdate:
        """Search-specific `tell` update computation. Returns state update."""
        # Reconstruct noise from last mean/std estimates
        scaled_noise = x - state.mean
        noise_1 = scaled_noise[::2]
        fit_1 = fitness[::2]
        fit_2 = fitness[1::2]
        elite_idx = jnp.minimum(fit_1, fit_2).argsort()[: self.elite_popsize]

        fitness_elite = jnp.concatenate([fit_1[elite_idx], fit_2[elite_idx]])
        fit_diff = fit_1[elite_idx] - fit_2[elite_idx]
        fit_diff_noise = noise_1[elite_idx] * fit_diff[:, None]

        grad_mean = (0.5 * fit_diff_noise).mean(axis=0)
        baseline = jnp.mean(fitness_elite)
        all_avg_scores = jnp.stack([fit_1[elite_idx], fit_2[elite_idx]]).sum(axis=0) / 2

        # Update sigma vector
        delta_sigma = (
            (jnp.expand_dims(all_avg_scores, axis=1) - baseline)
            * (noise_1[elite_idx] ** 2 - jnp.expand_dims(state.sigma**2, axis=0))
            / state.sigma
        ).mean(axis=0)
        return EvoUpdate(grad_mean=grad_mean, delta_sigma=delta_sigma)

    def apply_update_strategy(
        self,
        update: EvoUpdate,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """Search-specific `tell` update application. Returns updated state."""
        # Grad update using optimizer instance - decay lrate if desired
        grad_mean = jax.lax.pmean(update.grad_mean, "device")
        mean, opt_state = self.optimizer.step(
            state.mean, grad_mean, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        delta_sigma = jax.lax.pmean(update.delta_sigma, "device")
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
