import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


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
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        weights_prime = jnp.array(
            [
                jnp.log((self.popsize + 1) / 2) - jnp.log(i + 1)
                for i in range(self.popsize)
            ]
        )
        mu_eff = (jnp.sum(weights_prime[: self.elite_popsize]) ** 2) / jnp.sum(
            weights_prime[: self.elite_popsize] ** 2
        )
        mu_eff_minus = (
            jnp.sum(weights_prime[self.elite_popsize :]) ** 2
        ) / jnp.sum(weights_prime[self.elite_popsize :] ** 2)

        # lrates for rank-one and rank-μ C updates
        alpha_cov = 2
        c_1 = alpha_cov / ((self.num_dims + 1.3) ** 2 + mu_eff)
        c_mu = jnp.minimum(
            1 - c_1 - 1e-8,
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((self.num_dims + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        min_alpha = min(
            1 + c_1 / c_mu,
            1 + (2 * mu_eff_minus) / (mu_eff + 2),
            (1 - c_1 - c_mu) / (self.num_dims * c_mu),
        )
        positive_sum = jnp.sum(weights_prime[weights_prime > 0])
        negative_sum = jnp.sum(jnp.abs(weights_prime[weights_prime < 0]))
        weights = jnp.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        weights_truncated = weights.at[self.elite_popsize :].set(0)

        # lrate for cumulation of step-size control and rank-one update
        c_sigma = jnp.minimum(2 * self.popsize / self.num_dims, 1)
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
        mu_w = 1 / jnp.sum(weights_truncated ** 2)
        params = {
            "mu_eff": mu_eff,
            "c_1": c_1,
            "c_mu": c_mu,
            "c_m": 1.0,
            "c_sigma": c_sigma,
            "d_sigma": d_sigma,
            "c_c": c_c,
            "c_d": c_d,
            "chi_n": chi_n,
            "sigma_init": 0.065,
            "weights_truncated": weights_truncated,
            "mu_w": mu_w,
            "init_min": 0.0,
            "init_max": 0.0,
        }
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        # Initialize evolution paths & covariance matrix
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "p_sigma": jnp.zeros(self.num_dims),
            "sigma": params["sigma_init"],
            "mean": initialization,
            "M": jnp.zeros((self.num_dims, self.memory_size)),
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        x = sample(
            rng,
            state["mean"],
            state["sigma"],
            state["M"],
            self.num_dims,
            self.popsize,
            params["c_d"],
            state["gen_counter"],
        )
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean, z_k = update_mean(
            state["mean"], state["sigma"], sorted_solutions, params
        )
        p_sigma, norm_p_sigma = update_p_sigma(z_k, state["p_sigma"], params)
        M = update_M_matrix(state["M"], z_k, params)
        sigma = update_sigma(state["sigma"], norm_p_sigma, params)

        state["mean"] = mean
        state["p_sigma"] = p_sigma
        state["M"] = M
        state["sigma"] = sigma
        return state


def update_mean(
    mean: chex.Array,
    sigma: float,
    sorted_solutions: chex.Array,
    params: chex.ArrayTree,
) -> Tuple[chex.Array, chex.Array]:
    """Update mean of strategy."""
    z_k = sorted_solutions[:, 1:] - mean  # ~ N(0, σ^2 C)
    y_k = z_k / sigma  # ~ N(0, C)
    y_w = jnp.sum(y_k.T * params["weights_truncated"], axis=1)
    mean += params["c_m"] * sigma * y_w
    return mean, z_k


def update_p_sigma(
    z_k: chex.Array,
    p_sigma: chex.Array,
    params: chex.ArrayTree,
) -> Tuple[chex.Array, float]:
    """Update evolution path for covariance matrix."""
    z_w = jnp.sum(z_k.T * params["weights_truncated"], axis=1)
    p_sigma_new = (1 - params["c_sigma"]) * p_sigma + jnp.sqrt(
        params["c_sigma"] * (2 - params["c_sigma"]) * params["mu_eff"]
    ) * z_w
    norm_p_sigma = jnp.linalg.norm(p_sigma_new)
    return p_sigma_new, norm_p_sigma


def update_M_matrix(
    M: chex.Array, z_k: chex.Array, params: chex.ArrayTree
) -> chex.Array:
    """Update the M matrix."""
    weighted_elite = jnp.sum(
        jnp.array([w * z for w, z in zip(params["weights_truncated"], z_k)]),
        axis=0,
    )
    # Loop over individual memory components - this could be vectorized!
    for i in range(M.shape[1]):
        new_m = (1 - params["c_c"][i]) * M[:, i] + jnp.sqrt(
            params["mu_w"] * params["c_c"][i] * (2 - params["c_c"][i])
        ) * weighted_elite
        M = M.at[:, i].set(new_m)
    return M


def update_sigma(
    sigma: float, norm_p_sigma: float, params: chex.ArrayTree
) -> float:
    """Update stepsize sigma."""
    sigma_new = sigma * jnp.exp(
        (params["c_sigma"] / params["d_sigma"])
        * (norm_p_sigma / params["chi_n"] - 1)
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
