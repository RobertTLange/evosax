import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..core import GradientOptimizer, OptParams, OptState, exp_decay
from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    unroll_pert: chex.Array  # Noise perturb used in partial unroll multi times
    inner_step_counter: int  # Keep track of unner unroll steps for noise reset
    opt_state: OptState
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class EvoParams:
    opt_params: OptParams
    T: int = 100  # Total inner problem length
    K: int = 10  # Truncation length for partial unrolls
    sigma_init: float = 0.1
    sigma_decay: float = 0.999
    sigma_limit: float = 0.1
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class NoiseReuseES(Strategy):
    def __init__(
        self,
        population_size: int,
        pholder_params: chex.ArrayTree | chex.Array | None = None,
        opt_name: str = "adam",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_init: float = 0.03,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Noise-Reuse ES (Li et al., 2023).
        Reference: https://arxiv.org/pdf/2304.12180.pdf
        """
        super().__init__(population_size, pholder_params, mean_decay, **fitness_kwargs)
        assert not self.population_size & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "NoiseReuseES"

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

    def init_strategy(self, key: jax.Array, params: EvoParams) -> chex.ArrayTree:
        """`init` the evolution strategy."""
        initialization = jax.random.uniform(
            key,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            unroll_pert=jnp.zeros((self.population_size, self.num_dims)),
            opt_state=self.optimizer.init(params.opt_params),
            sigma=params.sigma_init,
            inner_step_counter=0,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: EvoState, params: EvoParams
    ) -> tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        # Generate antithetic perturbations
        # NOTE: Sample each ask call - only use when trajectory is reset
        pos_perts = (
            jax.random.normal(key, (self.population_size // 2, self.num_dims))
            * state.sigma
        )
        neg_perts = -pos_perts
        perts = jnp.concatenate([pos_perts, neg_perts], axis=0)
        unroll_pert = jax.lax.select(
            state.inner_step_counter == 0, perts, state.unroll_pert
        )
        # Add the perturbations from this unroll to the perturbation accumulators
        x = state.mean + unroll_pert
        return x, state.replace(unroll_pert=unroll_pert)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        theta_grad = jnp.mean(
            state.unroll_pert * fitness.reshape(-1, 1) / (state.sigma**2),
            axis=0,
        )
        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)
        inner_step_counter = state.inner_step_counter + params.K

        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        # Resample antithetic noise if done with inner problem
        reset = inner_step_counter >= params.T
        inner_step_counter = jax.lax.select(reset, 0, inner_step_counter)
        return state.replace(
            mean=mean,
            sigma=sigma,
            opt_state=opt_state,
            inner_step_counter=inner_step_counter,
        )
