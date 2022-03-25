import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class MA_ES(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.5):
        """MA-ES (Bayer & Sendhoff, 2017)
        Reference: https://www.honda-ri.de/pubs/pdf/3376.pdf
        """
        super().__init__(num_dims, popsize)
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.strategy_name = "MA_ES"

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

        params = {
            "mu_eff": mu_eff,
            "c_1": c_1,
            "c_mu": c_mu,
            "c_m": 1.0,
            "c_sigma": c_sigma,
            "d_sigma": d_sigma,
            "chi_n": chi_n,
            "sigma_init": 0.065,
            "weights_truncated": weights_truncated,
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
            "M": jnp.eye(self.num_dims),
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
        M = update_M_matrix(state["M"], state["p_sigma"], z_k, params)
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
    M: chex.Array, p_sigma: chex.Array, z_k: chex.Array, params: chex.ArrayTree
) -> chex.Array:
    """Update the M matrix."""
    rank_one = jnp.outer(p_sigma, p_sigma)
    rank_mu = jnp.sum(
        jnp.array(
            [
                w * jnp.outer(z, z)
                for w, z in zip(params["weights_truncated"], z_k)
            ]
        ),
        axis=0,
    )
    M_new = M @ (
        jnp.eye(M.shape[0])
        + params["c_1"] / 2 * (rank_one - jnp.eye(M.shape[0]))
        + params["c_mu"] / 2 * (rank_mu - jnp.eye(M.shape[0]))
    )
    return M_new


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
) -> chex.Array:
    """Jittable Gaussian Sample Helper."""
    z = jax.random.normal(rng, (n_dim, pop_size))  # ~ N(0, I)
    y = M.dot(z)  # ~ N(0, C)
    y = jnp.swapaxes(y, 1, 0)
    x = mean + sigma * y  # ~ N(m, σ^2 C)
    return x
