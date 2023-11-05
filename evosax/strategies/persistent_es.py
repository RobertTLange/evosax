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
    sigma: float
    pert_accum: chex.Array  # History of accum. noise perturb in partial unroll
    inner_step_counter: int  # Keep track of unner unroll steps for noise reset
    opt_state: OptState
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


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


class PersistentES(Strategy):
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
        """Persistent ES (Vicol et al., 2021).
        Reference: http://proceedings.mlr.press/v139/vicol21a.html
        Inspired by: http://proceedings.mlr.press/v139/vicol21a/vicol21a-supp.pdf
        """
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            mean_decay,
            n_devices,
            **fitness_kwargs
        )
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "PersistentES"

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
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            pert_accum=jnp.zeros((self.popsize, self.num_dims)),
            opt_state=self.optimizer.initialize(params.opt_params),
            sigma=params.sigma_init,
            inner_step_counter=0,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        # Generate antithetic perturbations
        pos_perts = (
            jax.random.normal(rng, (self.popsize // 2, self.num_dims))
            * state.sigma
        )
        neg_perts = -pos_perts
        perts = jnp.concatenate([pos_perts, neg_perts], axis=0)
        # Add the perturbations from this unroll to the perturbation accumulators
        pert_accum = state.pert_accum + perts
        x = state.mean + perts
        return x, state.replace(pert_accum=pert_accum)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        theta_grad = jnp.mean(
            state.pert_accum * fitness.reshape(-1, 1) / (state.sigma ** 2),
            axis=0,
        )
        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)
        inner_step_counter = state.inner_step_counter + params.K

        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        # Reset accumulated antithetic noise memory if done with inner problem
        reset = inner_step_counter >= params.T
        inner_step_counter = jax.numpy.where(reset, 0, inner_step_counter)
        pert_accum = jax.numpy.where(
            reset, jnp.zeros((self.popsize, self.num_dims)), state.pert_accum
        )
        return state.replace(
            mean=mean,
            sigma=sigma,
            opt_state=opt_state,
            pert_accum=pert_accum,
            inner_step_counter=inner_step_counter,
        )
