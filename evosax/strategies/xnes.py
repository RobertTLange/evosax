from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from .snes import get_snes_weights
from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    B: chex.Array
    noise: chex.Array
    lrate_sigma: float
    weights: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    lrate_mean: float = 1.0
    lrate_sigma_init: float = 0.1
    lrate_B: float = 0.1
    sigma_init: float = 1.0
    use_adasam: bool = False  # Adaptation sampling lrate sigma
    rho: float = 0.5  # Significance level adaptation sampling
    c_prime: float = 0.1  # Adaptation sampling step size
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class xNES(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Exponential Natural ES (Wierstra et al., 2014)
        Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        Inspired by: https://github.com/chanshing/xnes"""
        super().__init__(
            popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs
        )
        self.strategy_name = "xNES"

        # Set core kwargs es_params
        self.sigma_init = sigma_init

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolutionary strategy."""
        lrate_sigma = (9 + 3 * jnp.log(self.num_dims)) / (
            5 * jnp.sqrt(self.num_dims) * self.num_dims
        )
        rho = 0.5 - 1.0 / (3 * (self.num_dims + 1))
        params = EvoParams(
            lrate_sigma_init=lrate_sigma,
            lrate_B=lrate_sigma,
            rho=rho,
            sigma_init=self.sigma_init,
        )
        return params

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolutionary strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        weights = get_snes_weights(self.popsize)
        state = EvoState(
            mean=initialization,
            B=jnp.eye(self.num_dims) * params.sigma_init,
            sigma=params.sigma_init,
            noise=jnp.zeros((self.popsize, self.num_dims)),
            lrate_sigma=params.lrate_sigma_init,
            weights=weights,
            best_member=initialization,
        )

        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        noise = jax.random.normal(rng, (self.popsize, self.num_dims))

        def scale_orient(n, sigma, B):
            return sigma * B.T @ n

        scaled_noise = jax.vmap(scale_orient, in_axes=(0, None, None))(
            noise, state.sigma, state.B
        )
        x = state.mean + scaled_noise
        return x, state.replace(noise=noise)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        ranks = fitness.argsort()
        sorted_noise = state.noise[ranks]
        grad_mean = (state.weights * sorted_noise).sum(axis=0)

        def s_grad_m(weight, noise):
            return weight * (noise @ noise.T - jnp.eye(self.num_dims))

        grad_m = jax.vmap(s_grad_m, in_axes=(0, 0))(state.weights, sorted_noise).sum(
            axis=0
        )
        grad_sigma = jnp.trace(grad_m) / self.num_dims
        grad_B = grad_m - grad_sigma * jnp.eye(self.num_dims)

        mean = state.mean + params.lrate_mean * state.sigma * state.B @ grad_mean
        sigma = state.sigma * jnp.exp(state.lrate_sigma / 2 * grad_sigma)
        B = state.B * jnp.exp(params.lrate_B / 2 * grad_B)

        lrate_sigma = adaptation_sampling(
            state.lrate_sigma,
            params.lrate_sigma_init,
            mean,
            B,
            sigma,
            state.sigma,
            sorted_noise,
            params.c_prime,
            params.rho,
        )
        lrate_sigma = jax.lax.select(params.use_adasam, lrate_sigma, state.lrate_sigma)
        return state.replace(mean=mean, sigma=sigma, B=B, lrate_sigma=lrate_sigma)


def adaptation_sampling(
    lrate_sigma: float,
    lrate_sigma_init: float,
    mean: chex.Array,
    B: chex.Array,
    sigma: float,
    sigma_old: float,
    sorted_noise: chex.Array,
    c_prime: float,
    rho: float,
) -> float:
    """Adaptation sampling on sigma/std learning rate."""
    BB = B.T @ B
    A = sigma**2 * BB
    sigma_prime = sigma * jnp.sqrt(sigma / sigma_old)
    A_prime = sigma_prime**2 * BB

    # Probability ration and u-test - sorted order assumed for noise
    prob_0 = jax.scipy.stats.multivariate_normal.logpdf(sorted_noise, mean, A)
    prob_1 = jax.scipy.stats.multivariate_normal.logpdf(sorted_noise, mean, A_prime)
    w = jnp.exp(prob_1 - prob_0)
    popsize = sorted_noise.shape[0]
    n = jnp.sum(w)
    u = jnp.sum(w * (jnp.arange(popsize) + 0.5))
    u_mean = popsize * n / 2
    u_sigma = jnp.sqrt(popsize * n * (popsize + n + 1) / 12)
    cumulative = jax.scipy.stats.norm.cdf(u, loc=u_mean + 1e-10, scale=u_sigma + 1e-10)

    # Check test significance and update lrate
    lrate_sigma = jax.lax.select(
        cumulative < rho,
        (1 - c_prime) * lrate_sigma + c_prime * lrate_sigma_init,
        jnp.minimum(1, (1 - c_prime) * lrate_sigma),
    )
    return lrate_sigma
