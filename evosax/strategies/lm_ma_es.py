import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from .cma_es import get_cma_elite_weights
from flax import struct


@struct.dataclass
class EvoState:
    p_sigma: chex.Array
    M: chex.Array
    mean: chex.Array
    sigma: float
    c_c: chex.Array
    c_d: chex.Array
    weights_truncated: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    mu_eff: float
    c_1: float
    c_mu: float
    c_sigma: float
    d_sigma: float
    chi_n: float
    mu_w: float
    c_m: float = 1.0
    sigma_init: float = 0.065
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class LM_MA_ES(Strategy):
    def __init__(
        self,
        num_dims: int,
        popsize: int,
        elite_ratio: float = 0.5,
        memory_size: int = 10,
    ):
        """Limited Memory MA-ES (Loshchilov et al., 2017)
        Reference: https://arxiv.org/pdf/1705.06693.pdf
        """
        super().__init__(num_dims, popsize)
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.memory_size = memory_size
        self.strategy_name = "LM_MA_ES"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        _, weights_truncated, mu_eff, c_1, c_mu = get_cma_elite_weights(
            self.popsize, self.elite_popsize, self.num_dims
        )

        # lrate for cumulation of step-size control and rank-one update
        c_sigma = (mu_eff + 2) / (self.num_dims + mu_eff + 5)
        d_sigma = (
            1
            + 2
            * jnp.maximum(0, jnp.sqrt((mu_eff - 1) / (self.num_dims + 1)) - 1)
            + c_sigma
        )
        chi_n = jnp.sqrt(self.num_dims) * (
            1.0
            - (1.0 / (4.0 * self.num_dims))
            + 1.0 / (21.0 * (self.num_dims ** 2))
        )
        mu_w = 1 / jnp.sum(weights_truncated ** 2)
        params = EvoParams(
            mu_eff=mu_eff,
            c_1=c_1,
            c_mu=c_mu,
            c_sigma=c_sigma,
            d_sigma=d_sigma,
            chi_n=chi_n,
            mu_w=mu_w,
        )
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        _, weights_truncated, _, _, _ = get_cma_elite_weights(
            self.popsize, self.elite_popsize, self.num_dims
        )
        c_d = jnp.array(
            [1 / (1.5 ** i * self.num_dims) for i in range(self.memory_size)]
        )
        c_c = jnp.array(
            [
                self.popsize / (4 ** i * self.num_dims)
                for i in range(self.memory_size)
            ]
        )
        c_c = jnp.minimum(c_c, 1.99)

        # Initialize evolution paths & covariance matrix
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            p_sigma=jnp.zeros(self.num_dims),
            sigma=params.sigma_init,
            mean=initialization,
            M=jnp.zeros((self.num_dims, self.memory_size)),
            weights_truncated=weights_truncated,
            c_d=c_d,
            c_c=c_c,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        x = sample(
            rng,
            state.mean,
            state.sigma,
            state.M,
            self.num_dims,
            self.popsize,
            state.c_d,
            state.gen_counter,
        )
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean, z_k = update_mean(
            state.mean,
            state.sigma,
            sorted_solutions,
            params.c_m,
            state.weights_truncated,
        )
        p_sigma, norm_p_sigma = update_p_sigma(
            z_k,
            state.p_sigma,
            params.c_sigma,
            params.mu_eff,
            state.weights_truncated,
        )
        M = update_M_matrix(
            state.M,
            z_k,
            state.c_c,
            params.mu_w,
            state.weights_truncated,
        )
        sigma = update_sigma(
            state.sigma,
            norm_p_sigma,
            params.c_sigma,
            params.d_sigma,
            params.chi_n,
        )
        return state.replace(mean=mean, p_sigma=p_sigma, M=M, sigma=sigma)


def update_mean(
    mean: chex.Array,
    sigma: float,
    sorted_solutions: chex.Array,
    c_m: float,
    weights_truncated: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Update mean of strategy."""
    z_k = sorted_solutions[:, 1:] - mean  # ~ N(0, σ^2 C)
    y_k = z_k / sigma  # ~ N(0, C)
    y_w = jnp.sum(y_k.T * weights_truncated, axis=1)
    mean += c_m * sigma * y_w
    return mean, z_k


def update_p_sigma(
    z_k: chex.Array,
    p_sigma: chex.Array,
    c_sigma: float,
    mu_eff: float,
    weights_truncated: chex.Array,
) -> Tuple[chex.Array, float]:
    """Update evolution path for covariance matrix."""
    z_w = jnp.sum(z_k.T * weights_truncated, axis=1)
    p_sigma_new = (1 - c_sigma) * p_sigma + jnp.sqrt(
        c_sigma * (2 - c_sigma) * mu_eff
    ) * z_w
    norm_p_sigma = jnp.linalg.norm(p_sigma_new)
    return p_sigma_new, norm_p_sigma


def update_M_matrix(
    M: chex.Array,
    z_k: chex.Array,
    c_c: chex.Array,
    mu_w: float,
    weights_truncated: chex.Array,
) -> chex.Array:
    """Update the M matrix."""
    weighted_elite = jnp.sum(
        jnp.array([w * z for w, z in zip(weights_truncated, z_k)]),
        axis=0,
    )
    # Loop over individual memory components - this could be vectorized!
    for i in range(M.shape[1]):
        new_m = (1 - c_c[i]) * M[:, i] + jnp.sqrt(
            mu_w * c_c[i] * (2 - c_c[i])
        ) * weighted_elite
        M = M.at[:, i].set(new_m)
    return M


def update_sigma(
    sigma: float,
    norm_p_sigma: float,
    c_sigma: float,
    d_sigma: float,
    chi_n: float,
) -> float:
    """Update stepsize sigma."""
    sigma_new = sigma * jnp.exp(
        (c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1)
    )
    return sigma_new


def sample(
    rng: chex.PRNGKey,
    mean: chex.Array,
    sigma: float,
    M: chex.Array,
    n_dim: int,
    pop_size: int,
    c_d: chex.Array,
    gen_counter: int,
) -> chex.Array:
    """Jittable Gaussian Sample Helper."""
    z = jax.random.normal(rng, (n_dim, pop_size))  # ~ N(0, I)
    for j in range(M.shape[1]):
        update_bool = gen_counter > j
        new_z = (1 - c_d[j]) * z + (c_d[j] * M[:, j])[:, jnp.newaxis] * (
            M[:, j][:, jnp.newaxis] * z
        )
        z = jax.lax.select(update_bool, new_z, z)
    z = jnp.swapaxes(z, 1, 0)
    x = mean + sigma * z  # ~ N(m, σ^2 C)
    return x
