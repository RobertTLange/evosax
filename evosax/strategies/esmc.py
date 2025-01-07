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
    sigma_init: float = 0.03
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    sigma_lrate: float = 0.2  # Learning rate for std
    sigma_max_change: float = 0.2  # Clip adaptive sigma to 20%
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class ESMC(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        opt_name: str = "adam",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_init: float = 0.03,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """ESMC (Merchant et al., 2021)
        Reference: https://proceedings.mlr.press/v139/merchant21a.html
        """
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            mean_decay,
            n_devices,
            **fitness_kwargs
        )
        assert self.popsize & 1, "Population size must be odd"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "ESMC"

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
        z = jnp.concatenate(
            [jnp.zeros((1, self.num_dims)), z_plus, -1.0 * z_plus]
        )
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
        bline_fitness = fitness[0]
        noise = noise[1:]
        fitness = fitness[1:]
        noise_1 = noise[: int((self.popsize - 1) / 2)]
        fit_1 = fitness[: int((self.popsize - 1) / 2)]
        fit_2 = fitness[int((self.popsize - 1) / 2) :]
        fit_diff = jnp.minimum(fit_1, bline_fitness) - jnp.minimum(
            fit_2, bline_fitness
        )
        fit_diff_noise = jnp.dot(noise_1.T, fit_diff)
        theta_grad = 1.0 / int((self.popsize - 1) / 2) * fit_diff_noise
        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)
        sigma = state.sigma * params.sigma_decay
        sigma = jnp.maximum(sigma, params.sigma_limit)
        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state)
