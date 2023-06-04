from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..strategy import Strategy


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
    z: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    c_sigma: float
    d_sigma: float
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
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        elite_ratio: float = 0.5,
        memory_size: int = 10,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Limited Memory MA-ES (Loshchilov et al., 2017)
        Reference: https://arxiv.org/pdf/1705.06693.pdf
        """
        super().__init__(
            popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs
        )
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
        self.memory_size = memory_size
        self.strategy_name = "LM_MA_ES"

        # Set core kwargs es_params
        self.sigma_init = sigma_init

        # Robustness for int32 - squaring in hyperparameter calculations
        self.max_dims_sq = jnp.minimum(self.num_dims, 40000)

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        w_hat = jnp.array(
            [
                jnp.log(self.elite_popsize + 0.5) - jnp.log(i)
                for i in range(1, self.elite_popsize + 1)
            ]
        )
        weights_truncated = w_hat / jnp.sum(w_hat)
        # lrate for cumulation of step-size control and rank-one update
        c_sigma = (2 * self.popsize) / self.num_dims
        d_sigma = 2
        mu_w = 1 / jnp.sum(jnp.square(weights_truncated))
        params = EvoParams(
            c_sigma=c_sigma,
            d_sigma=d_sigma,
            mu_w=mu_w,
            sigma_init=self.sigma_init,
        )
        return params

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        w_hat = jnp.array(
            [
                jnp.log(self.elite_popsize + 0.5) - jnp.log(i)
                for i in range(1, self.elite_popsize + 1)
            ]
        )
        weights_truncated = w_hat / jnp.sum(w_hat)
        c_d = jnp.array(
            [1 / (1.5**i * self.num_dims) for i in range(self.memory_size)]
        )
        c_c = jnp.array(
            [self.popsize / (4**i * self.num_dims) for i in range(self.memory_size)]
        )

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
            M=jnp.zeros((self.memory_size, self.num_dims)),
            weights_truncated=weights_truncated,
            c_d=c_d,
            c_c=c_c,
            best_member=initialization,
            z=jnp.zeros((self.popsize, self.num_dims)),
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        x, z = sample(
            rng,
            state.mean,
            state.sigma,
            state.M,
            self.num_dims,
            self.popsize,
            state.c_d,
            state.gen_counter,
        )
        return x, state.replace(z=z)

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
        concat_z_f = jnp.hstack([jnp.expand_dims(fitness, 1), state.z])
        sorted_zvectors = concat_z_f[concat_z_f[:, 0].argsort()]
        sorted_z = sorted_zvectors[:, 1:]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean = update_mean(
            state.mean,
            sorted_solutions,
            self.elite_popsize,
            params.c_m,
            state.weights_truncated,
        )
        p_sigma, norm_p_sigma, wz = update_p_sigma(
            sorted_z,
            self.elite_popsize,
            state.p_sigma,
            params.c_sigma,
            params.mu_w,
            state.weights_truncated,
        )
        M = update_M_matrix(
            state.M,
            wz,
            state.c_c,
            params.mu_w,
        )
        sigma = update_sigma(
            state.sigma,
            norm_p_sigma,
            params.c_sigma,
            params.d_sigma,
            self.num_dims,
        )
        return state.replace(mean=mean, p_sigma=p_sigma, M=M, sigma=sigma)


def update_mean(
    mean: chex.Array,
    sorted_solutions: chex.Array,
    elite_popsize: int,
    c_m: float,
    weights_truncated: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Update mean of strategy."""
    y_k = sorted_solutions[:elite_popsize, 1:] - mean  # ~ N(0, σ^2 C)
    G_m = weights_truncated.T @ y_k
    mean += c_m * G_m
    return mean


def update_p_sigma(
    z_k: chex.Array,
    elite_popsize: int,
    p_sigma: chex.Array,
    c_sigma: float,
    mu_w: float,
    weights_truncated: chex.Array,
) -> Tuple[chex.Array, float]:
    """Update evolution path for covariance matrix."""
    wz = weights_truncated.T @ (z_k[:elite_popsize, :])
    p_sigma_new = (1 - c_sigma) * p_sigma + jnp.sqrt(
        mu_w * c_sigma * (2 - c_sigma)
    ) * wz
    norm_p_sigma = jnp.linalg.norm(p_sigma_new)
    return p_sigma_new, norm_p_sigma, wz


def update_M_matrix(
    M: chex.Array,
    wz: chex.Array,
    c_c: chex.Array,
    mu_w: float,
) -> chex.Array:
    """Update the M matrix."""
    # Loop over individual memory components - this could be vectorized!
    for i in range(M.shape[0]):
        new_m = (1 - c_c[i]) * M[i, :] + jnp.sqrt(mu_w * c_c[i] * (2 - c_c[i])) * wz
        M = M.at[i, :].set(new_m)
    return M


def update_sigma(
    sigma: float,
    norm_p_sigma: float,
    c_sigma: float,
    d_sigma: float,
    n: int,
) -> float:
    """Update stepsize sigma."""
    sigma_new = sigma * jnp.exp(
        (c_sigma / d_sigma) * (jnp.square(norm_p_sigma) / n - 1)
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
) -> tuple[chex.Array, chex.Array]:
    """Jittable Gaussian Sample Helper."""
    z = jax.random.normal(rng, (pop_size, n_dim))  # ~ N(0, I)
    d = jnp.copy(z)
    for j in range(M.shape[0]):
        update_bool = gen_counter > j
        new_d = (1 - c_d[j]) * d + c_d[j] * jnp.outer(jnp.dot(d, M[j, :]), M[j, :])
        d = jax.lax.select(update_bool, new_d, d)
    x = mean + sigma * d  # ~ N(m, σ^2 C)
    return x, z
