import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy
from ..types import Fitness, Population, Solution
from .cma_es import get_cma_elite_weights


@struct.dataclass
class State:
    p_sigma: jax.Array
    M: jax.Array
    mean: jax.Array
    sigma: float
    weights_truncated: jax.Array
    best_member: jax.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class Params:
    mu_eff: float
    c_1: float
    c_mu: float
    c_sigma: float
    d_sigma: float
    chi_n: float
    c_m: float = 1.0
    sigma_init: float = 0.065
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class MA_ES(Strategy):
    def __init__(
        self,
        population_size: int,
        solution: Solution,
        elite_ratio: float = 0.5,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """MA-ES (Bayer & Sendhoff, 2017)
        Reference: https://www.honda-ri.de/pubs/pdf/3376.pdf
        """
        super().__init__(population_size, solution, mean_decay, **fitness_kwargs)
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_population_size = max(
            1, int(self.population_size * self.elite_ratio)
        )
        self.strategy_name = "MA_ES"

        # Set core kwargs params
        self.sigma_init = sigma_init

        # Robustness for int32 - squaring in hyperparameter calculations
        self.max_dims_sq = jnp.minimum(self.num_dims, 40000)

    @property
    def params_strategy(self) -> Params:
        """Return default parameters of evolution strategy."""
        _, _, mu_eff, c_1, c_mu = get_cma_elite_weights(
            self.population_size,
            self.elite_population_size,
            self.num_dims,
            self.max_dims_sq,
        )

        # lrate for cumulation of step-size control and rank-one update
        c_sigma = (mu_eff + 2) / (self.num_dims + mu_eff + 5)
        d_sigma = (
            1
            + 2 * jnp.maximum(0, jnp.sqrt((mu_eff - 1) / (self.num_dims + 1)) - 1)
            + c_sigma
        )

        chi_n = jnp.sqrt(self.num_dims) * (
            1.0 - (1.0 / (4.0 * self.num_dims)) + 1.0 / (21.0 * (self.max_dims_sq**2))
        )

        params = Params(
            mu_eff=mu_eff,
            c_1=c_1,
            c_mu=c_mu,
            c_sigma=c_sigma,
            d_sigma=d_sigma,
            chi_n=chi_n,
            sigma_init=self.sigma_init,
        )
        return params

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """`init` the evolution strategy."""
        _, weights_truncated, _, _, _ = get_cma_elite_weights(
            self.population_size,
            self.elite_population_size,
            self.num_dims,
            self.max_dims_sq,
        )
        # Initialize evolution paths & covariance matrix
        initialization = jax.random.uniform(
            key,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = State(
            p_sigma=jnp.zeros(self.num_dims),
            sigma=params.sigma_init,
            mean=initialization,
            M=jnp.eye(self.num_dims),
            weights_truncated=weights_truncated,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[jax.Array, State]:
        """`ask` for new parameter candidates to evaluate next."""
        x = sample(
            key,
            state.mean,
            state.sigma,
            state.M,
            self.num_dims,
            self.population_size,
        )
        return x, state

    def tell_strategy(
        self,
        x: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
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
            state.p_sigma,
            z_k,
            params.c_1,
            params.c_mu,
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
    mean: jax.Array,
    sigma: float,
    sorted_solutions: jax.Array,
    c_m: float,
    weights_truncated: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Update mean of strategy."""
    z_k = sorted_solutions[:, 1:] - mean  # ~ N(0, σ^2 C)
    y_k = z_k / sigma  # ~ N(0, C)
    y_w = jnp.sum(y_k.T * weights_truncated, axis=1)
    mean += c_m * sigma * y_w
    return mean, z_k


def update_p_sigma(
    z_k: jax.Array,
    p_sigma: jax.Array,
    c_sigma: float,
    mu_eff: float,
    weights_truncated: jax.Array,
) -> tuple[jax.Array, float]:
    """Update evolution path for covariance matrix."""
    z_w = jnp.sum(z_k.T * weights_truncated, axis=1)
    p_sigma_new = (1 - c_sigma) * p_sigma + jnp.sqrt(
        c_sigma * (2 - c_sigma) * mu_eff
    ) * z_w
    norm_p_sigma = jnp.linalg.norm(p_sigma_new)
    return p_sigma_new, norm_p_sigma


def update_M_matrix(
    M: jax.Array,
    p_sigma: jax.Array,
    z_k: jax.Array,
    c_1: float,
    c_mu: float,
    weights_truncated: jax.Array,
) -> jax.Array:
    """Update the M matrix."""
    rank_one = jnp.outer(p_sigma, p_sigma)
    rank_mu = jnp.sum(
        jnp.array([w * jnp.outer(z, z) for w, z in zip(weights_truncated, z_k)]),
        axis=0,
    )
    M_new = M @ (
        jnp.eye(M.shape[0])
        + c_1 / 2 * (rank_one - jnp.eye(M.shape[0]))
        + c_mu / 2 * (rank_mu - jnp.eye(M.shape[0]))
    )
    return M_new


def update_sigma(
    sigma: float,
    norm_p_sigma: float,
    c_sigma: float,
    d_sigma: float,
    chi_n: float,
) -> float:
    """Update stepsize sigma."""
    sigma_new = sigma * jnp.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))
    return sigma_new


def sample(
    key: jax.Array,
    mean: jax.Array,
    sigma: float,
    M: jax.Array,
    n_dim: int,
    pop_size: int,
) -> jax.Array:
    """Jittable Gaussian Sample Helper."""
    z = jax.random.normal(key, (n_dim, pop_size))  # ~ N(0, I)
    y = M.dot(z)  # ~ N(0, C)
    y = jnp.swapaxes(y, 1, 0)
    x = mean + sigma * y  # ~ N(m, σ^2 C)
    return x
