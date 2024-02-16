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
    sigma_init: float = 0.04
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class OpenES(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        use_antithetic_sampling: bool = True,
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
        """OpenAI-ES (Salimans et al. (2017)
        Reference: https://arxiv.org/pdf/1703.03864.pdf
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(
            popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs
        )
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "OpenES"
        self.use_antithetic_sampling = use_antithetic_sampling

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
        if self.use_antithetic_sampling:
            z_plus = jax.random.normal(
                rng,
                (int(self.popsize / 2), self.num_dims),
            )
            z = jnp.concatenate([z_plus, -1.0 * z_plus])
        else:
            z = jax.random.normal(rng, (self.popsize, self.num_dims))
        x = state.mean + state.sigma * z
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # Reconstruct noise from last mean/std estimates
        noise = (x - state.mean) / state.sigma
        theta_grad = 1.0 / (self.popsize * state.sigma) * jnp.dot(noise.T, fitness)

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)
        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state)
