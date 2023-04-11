from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from functools import partial
from flax import struct
from ..strategy import Strategy
from ..core import GradientOptimizer, OptState, OptParams, exp_decay
from ..utils import get_best_fitness_member


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    opt_state: OptState
    grad_subspace: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    opt_params: OptParams
    sigma_init: float = 0.03
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    alpha: float = 0.5
    beta: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class GuidedES(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        subspace_dims: int = 1,  # k param in example notebook
        opt_name: str = "sgd",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_init: float = 0.03,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float],
    ):
        """Guided ES (Maheswaranathan et al., 2018)
        Reference: https://arxiv.org/abs/1806.10230
        Note that there are a couple of JAX-based adaptations:
        """
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            mean_decay,
            n_devices,
            **fitness_kwargs,
        )
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        assert (
            subspace_dims <= self.num_dims
        ), "Subspace has to be smaller than optimization dims."
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.subspace_dims = min(subspace_dims, self.num_dims)
        if self.subspace_dims < subspace_dims:
            print(
                "Subspace has to be smaller than optimization dims. Set to"
                f" {self.subspace_dims} instead of {subspace_dims}."
            )
        self.strategy_name = "GuidedES"

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
        return EvoParams(opt_params=self.optimizer.default_params)

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        rng_init, rng_sub = jax.random.split(rng)
        initialization = jax.random.uniform(
            rng_init,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )

        grad_subspace = jax.random.normal(
            rng_sub, (self.subspace_dims, self.num_dims)
        )

        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init,
            opt_state=self.optimizer.initialize(params.opt_params),
            grad_subspace=grad_subspace,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        a = state.sigma * jnp.sqrt(params.alpha / self.num_dims)
        c = state.sigma * jnp.sqrt((1.0 - params.alpha) / self.subspace_dims)
        key_full, key_sub = jax.random.split(rng, 2)
        eps_full = jax.random.normal(
            key_full, shape=(self.num_dims, int(self.popsize / 2))
        )
        eps_subspace = jax.random.normal(
            key_sub, shape=(self.subspace_dims, int(self.popsize / 2))
        )
        Q, _ = jnp.linalg.qr(state.grad_subspace)
        # Antithetic sampling of noise
        z_plus = a * eps_full + c * jnp.dot(Q, eps_subspace)
        z_plus = jnp.swapaxes(z_plus, 0, 1)
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.mean + z
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: Optional[EvoParams] = None,
        gradient: Optional[chex.Array] = None,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Flatten params if using param reshaper for ES update
        if self.use_param_reshaper:
            x = self.param_reshaper.flatten(x)

        # Perform fitness reshaping inside of strategy tell call (if desired)
        fitness_re = self.fitness_shaper.apply(x, fitness)

        # Reconstruct noise from last mean/std estimates
        noise = (x - state.mean) / state.sigma
        noise_1 = noise[: int(self.popsize / 2)]
        fit_1 = fitness_re[: int(self.popsize / 2)]
        fit_2 = fitness_re[int(self.popsize / 2) :]
        fit_diff = fit_1 - fit_2
        fit_diff_noise = jnp.dot(noise_1.T, fit_diff)
        theta_grad = (params.beta / self.popsize) * fit_diff_noise

        # Add grad FIFO-style to subspace archive (only if provided else FD)
        grad_subspace = jnp.zeros((self.subspace_dims, self.num_dims))
        grad_subspace = grad_subspace.at[:-1, :].set(state.grad_subspace[1:, :])
        if gradient is not None:
            grad_subspace = grad_subspace.at[-1, :].set(gradient)
        else:
            grad_subspace = grad_subspace.at[-1, :].set(theta_grad)
        state = state.replace(grad_subspace=grad_subspace)

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        # Update lrate and standard deviation based on min and decay
        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        state = state.replace(mean=mean, sigma=sigma, opt_state=opt_state)

        # Check if there is a new best member & update trackers
        best_member, best_fitness = get_best_fitness_member(x, fitness, state)
        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        )
