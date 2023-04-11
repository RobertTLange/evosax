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
    opt_state: OptState
    grad_subspace: chex.Array
    alpha: float
    UUT: chex.Array
    UUT_ort: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    opt_params: OptParams
    sigma_init: float = 0.03
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    grad_decay: float = 0.99
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class ASEBO(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        subspace_dims: int = 50,
        opt_name: str = "adam",
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
        """ASEBO (Choromanski et al., 2019)
        Reference: https://arxiv.org/abs/1903.04268
        Note that there are a couple of JAX-based adaptations:
        1. We always sample a fixed population size per generation
        2. We keep a fixed archive of gradients to estimate the subspace
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
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.subspace_dims = min(subspace_dims, self.num_dims)
        if self.subspace_dims < subspace_dims:
            print(
                "Subspace has to be smaller than optimization dims. Set to"
                f" {self.subspace_dims} instead of {subspace_dims}."
            )
        self.strategy_name = "ASEBO"

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

        grad_subspace = jnp.zeros((self.subspace_dims, self.num_dims))

        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init,
            opt_state=self.optimizer.initialize(params.opt_params),
            grad_subspace=grad_subspace,
            alpha=1.0,
            UUT=jnp.zeros((self.num_dims, self.num_dims)),
            UUT_ort=jnp.zeros((self.num_dims, self.num_dims)),
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        X = state.grad_subspace
        X -= jnp.mean(X, axis=0)
        U, S, Vt = jnp.linalg.svd(X, full_matrices=False)

        def svd_flip(u, v):
            # columns of u, rows of v
            max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
            signs = jnp.sign(u[max_abs_cols, jnp.arange(u.shape[1])])
            u *= signs
            v *= signs[:, jnp.newaxis]
            return u, v

        U, Vt = svd_flip(U, Vt)
        U = Vt[: int(self.popsize / 2)]
        UUT = jnp.matmul(U.T, U)

        U_ort = Vt[int(self.popsize / 2) :]
        UUT_ort = jnp.matmul(U_ort.T, U_ort)

        subspace_ready = state.gen_counter > self.subspace_dims

        UUT = jax.lax.select(
            subspace_ready, UUT, jnp.zeros((self.num_dims, self.num_dims))
        )
        cov = (
            state.sigma * (state.alpha / self.num_dims) * jnp.eye(self.num_dims)
            + ((1 - state.alpha) / int(self.popsize / 2)) * UUT
        )
        chol = jnp.linalg.cholesky(cov)
        noise = jax.random.normal(rng, (self.num_dims, int(self.popsize / 2)))
        z_plus = jnp.swapaxes(chol @ noise, 0, 1)
        z_plus /= jnp.linalg.norm(z_plus, axis=-1)[:, jnp.newaxis]
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.mean + z
        return x, state.replace(UUT=UUT, UUT_ort=UUT_ort)

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
        noise_1 = noise[: int(self.popsize / 2)]
        fit_1 = fitness[: int(self.popsize / 2)]
        fit_2 = fitness[int(self.popsize / 2) :]
        fit_diff_noise = jnp.dot(noise_1.T, fit_1 - fit_2)
        theta_grad = 1.0 / 2.0 * fit_diff_noise

        alpha = jnp.linalg.norm(
            jnp.dot(theta_grad, state.UUT_ort)
        ) / jnp.linalg.norm(jnp.dot(theta_grad, state.UUT))
        subspace_ready = state.gen_counter > self.subspace_dims
        alpha = jax.lax.select(subspace_ready, alpha, 1.0)

        # Add grad FIFO-style to subspace archive (only if provided else FD)
        grad_subspace = jnp.zeros((self.subspace_dims, self.num_dims))
        grad_subspace = grad_subspace.at[:-1, :].set(state.grad_subspace[1:, :])
        grad_subspace = grad_subspace.at[-1, :].set(theta_grad)
        state = state.replace(grad_subspace=grad_subspace)

        # Normalize gradients by norm / num_dims
        theta_grad /= jnp.linalg.norm(theta_grad) / self.num_dims + 1e-8

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        # Update lrate and standard deviation based on min and decay
        sigma = state.sigma * params.sigma_decay
        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        return state.replace(
            mean=mean, sigma=sigma, opt_state=opt_state, alpha=alpha
        )
