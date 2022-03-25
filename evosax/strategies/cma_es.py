import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from ..utils.eigen_decomp import full_eigen_decomp


class CMA_ES(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.5):
        """CMA-ES (e.g. Hansen, 2016)
        Reference: https://arxiv.org/abs/1604.00772
        Inspired by: https://github.com/CyberAgentAILab/cmaes"""
        super().__init__(num_dims, popsize)
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.strategy_name = "CMA_ES"

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
        c_c = (4 + mu_eff / self.num_dims) / (
            self.num_dims + 4 + 2 * mu_eff / self.num_dims
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
            "c_c": c_c,
            "chi_n": chi_n,
            "weights": weights,
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
            "p_c": jnp.zeros(self.num_dims),
            "sigma": params["sigma_init"],
            "mean": initialization,
            "C": jnp.eye(self.num_dims),
            "D": None,
            "B": None,
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        C, B, D = full_eigen_decomp(state["C"], state["B"], state["D"])
        x = sample(
            rng,
            state["mean"],
            state["sigma"],
            B,
            D,
            self.num_dims,
            self.popsize,
        )
        state["C"], state["B"], state["D"] = C, B, D
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
        y_k, y_w, mean = update_mean(
            state["mean"], state["sigma"], sorted_solutions, params
        )

        p_sigma, C_2, C, B, D = update_p_sigma(
            state["C"], state["B"], state["D"], state["p_sigma"], y_w, params
        )

        p_c, norm_p_sigma, h_sigma = update_p_c(
            mean, p_sigma, state["p_c"], state["gen_counter"] + 1, y_w, params
        )

        C = update_covariance(mean, p_c, C, y_k, h_sigma, C_2, params)
        sigma = update_sigma(state["sigma"], norm_p_sigma, params)

        state["mean"] = mean
        state["p_sigma"] = p_sigma
        state["C"] = C
        state["B"] = B
        state["D"] = D
        state["p_c"] = p_c
        state["sigma"] = sigma
        return state


def update_mean(
    mean: chex.Array,
    sigma: float,
    sorted_solutions: chex.Array,
    params: chex.ArrayTree,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Update mean of strategy."""
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = (x_k - mean) / sigma  # ~ N(0, C)
    y_w = jnp.sum(y_k.T * params["weights_truncated"], axis=1)
    mean += params["c_m"] * sigma * y_w
    return y_k, y_w, mean


def update_p_sigma(
    C: chex.Array,
    B: chex.Array,
    D: chex.Array,
    p_sigma: chex.Array,
    y_w: chex.Array,
    params: chex.ArrayTree,
) -> Tuple[chex.Array, chex.Array, chex.Array, None, None]:
    """Update evolution path for covariance matrix."""
    C, B, D = full_eigen_decomp(C, B, D)
    C_2 = B.dot(jnp.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
    p_sigma_new = (1 - params["c_sigma"]) * p_sigma + jnp.sqrt(
        params["c_sigma"] * (2 - params["c_sigma"]) * params["mu_eff"]
    ) * C_2.dot(y_w)
    _B, _D = None, None
    return p_sigma_new, C_2, C, _B, _D


def update_p_c(
    mean: chex.Array,
    p_sigma: chex.Array,
    p_c: chex.Array,
    gen_counter: int,
    y_w: chex.Array,
    params: chex.ArrayTree,
) -> Tuple[chex.Array, float, float]:
    """Update evolution path for sigma/stepsize."""
    norm_p_sigma = jnp.linalg.norm(p_sigma)
    h_sigma_cond_left = norm_p_sigma / jnp.sqrt(
        1 - (1 - params["c_sigma"]) ** (2 * (gen_counter))
    )
    h_sigma_cond_right = (1.4 + 2 / (mean.shape[0] + 1)) * params["chi_n"]
    h_sigma = 1.0 * (h_sigma_cond_left < h_sigma_cond_right)
    p_c_new = (1 - params["c_c"]) * p_c + h_sigma * jnp.sqrt(
        params["c_c"] * (2 - params["c_c"]) * params["mu_eff"]
    ) * y_w
    return p_c_new, norm_p_sigma, h_sigma


def update_covariance(
    mean: chex.Array,
    p_c: chex.Array,
    C: chex.Array,
    y_k: chex.Array,
    h_sigma: float,
    C_2: chex.Array,
    params: chex.ArrayTree,
) -> chex.Array:
    """Update cov. matrix estimator using rank 1 + μ updates."""
    w_io = params["weights"] * jnp.where(
        params["weights"] >= 0,
        1,
        mean.shape[0] / (jnp.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + 1e-20),
    )
    delta_h_sigma = (1 - h_sigma) * params["c_c"] * (2 - params["c_c"])
    rank_one = jnp.outer(p_c, p_c)
    rank_mu = jnp.sum(
        jnp.array([w * jnp.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
    )
    C = (
        (
            1
            + params["c_1"] * delta_h_sigma
            - params["c_1"]
            - params["c_mu"] * jnp.sum(params["weights"])
        )
        * C
        + params["c_1"] * rank_one
        + params["c_mu"] * rank_mu
    )
    return C


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
    B: chex.Array,
    D: chex.Array,
    n_dim: int,
    pop_size: int,
) -> chex.Array:
    """Jittable Gaussian Sample Helper."""
    z = jax.random.normal(rng, (n_dim, pop_size))  # ~ N(0, I)
    y = B.dot(jnp.diag(D)).dot(z)  # ~ N(0, C)
    y = jnp.swapaxes(y, 1, 0)
    x = mean + sigma * y  # ~ N(m, σ^2 C)
    return x
