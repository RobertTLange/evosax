import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class xNES(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Exponential Natural ES (Wierstra et al., 2014)
        Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        Inspired by: https://github.com/chanshing/xnes"""
        super().__init__(num_dims, popsize)
        self.strategy_name = "xNES"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolutionary strategy."""
        params = {
            "eta_mean": 1.0,
            "use_adaptive_sampling": False,
            "use_fitness_shaping": True,
            "eta_sigma_init": 3
            * (3 + jnp.log(self.num_dims))
            * (1.0 / (5 * self.num_dims * jnp.sqrt(self.num_dims))),
            "eta_bmat": 3
            * (3 + jnp.log(self.num_dims))
            * (1.0 / (5 * self.num_dims * jnp.sqrt(self.num_dims))),
            "init_min": 0.0,
            "init_max": 0.0,
        }
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.Array
    ) -> chex.ArrayTree:
        """`initialize` the evolutionary strategy."""
        amat = jnp.eye(self.num_dims)
        sigma = abs(jax.scipy.linalg.det(amat)) ** (1.0 / self.num_dims)
        bmat = amat * (1.0 / sigma)
        # Utility helper for fitness shaping - doesn't work without?!
        a = jnp.log(1 + 0.5 * self.popsize)
        utilities = jnp.array(
            [jnp.maximum(0, a - jnp.log(k)) for k in range(1, self.popsize + 1)]
        )
        utilities /= jnp.sum(utilities)
        utilities -= 1.0 / self.popsize  # broadcast
        utilities = utilities[::-1]  # ascending order

        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "mean": initialization,
            "sigma": sigma,
            "sigma_old": sigma,
            "amat": amat,
            "bmat": bmat,
            "noise": jnp.zeros((self.popsize, self.num_dims)),
            "eta_sigma": params["eta_sigma_init"],
            "utilities": utilities,
        }

        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        noise = jax.random.normal(rng, (self.popsize, self.num_dims))
        x = state["mean"] + state["sigma"] * jnp.dot(noise, state["bmat"])
        state["noise"] = noise
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # By default the xNES maximizes the objective
        fitness_re = -fitness
        isort = fitness_re.argsort()
        sorted_fitness = fitness_re[isort]
        sorted_noise = state["noise"][isort]
        sorted_candidates = x[isort]
        fitness_shaped = jax.lax.select(
            params["use_fitness_shaping"], state["utilities"], sorted_fitness
        )

        use_adasam = jnp.logical_and(
            params["use_adaptive_sampling"], state["gen_counter"] > 1
        )  # sigma_old must be available
        state["eta_sigma"] = jax.lax.select(
            use_adasam,
            self.adaptive_sampling(
                state["eta_sigma"],
                state["mean"],
                state["sigma"],
                state["bmat"],
                state["sigma_old"],
                sorted_candidates,
                params["eta_sigma_init"],
            ),
            state["eta_sigma"],
        )

        dj_delta = jnp.dot(fitness_shaped, sorted_noise)
        dj_mmat = (
            jnp.dot(
                sorted_noise.T,
                sorted_noise * fitness_shaped.reshape(self.popsize, 1),
            )
            - jnp.sum(fitness_shaped) * jnp.eye(self.num_dims)
        )
        dj_sigma = jnp.trace(dj_mmat) * (1.0 / self.num_dims)
        dj_bmat = dj_mmat - dj_sigma * jnp.eye(self.num_dims)

        state["sigma_old"] = state["sigma"]
        state["mean"] += (
            params["eta_mean"]
            * state["sigma"]
            * jnp.dot(state["bmat"], dj_delta)
        )
        state["sigma"] = state["sigma_old"] * jnp.exp(
            0.5 * state["eta_sigma"] * dj_sigma
        )
        state["bmat"] = jnp.dot(
            state["bmat"],
            jax.scipy.linalg.expm(0.5 * params["eta_bmat"] * dj_bmat),
        )
        return state

    def adaptive_sampling(
        self,
        eta_sigma: float,
        mu: chex.Array,
        sigma: float,
        bmat: chex.Array,
        sigma_old: float,
        z_try: chex.Array,
        eta_sigma_init: float,
    ) -> float:
        """Adaptation sampling."""
        c = 0.1
        rho = 0.5 - 1.0 / (3 * (self.num_dims + 1))  # empirical

        bbmat = jnp.dot(bmat.T, bmat)
        cov = sigma ** 2 * bbmat
        sigma_ = sigma * jnp.sqrt(sigma * (1.0 / sigma_old))  # increase by 1.5
        cov_ = sigma_ ** 2 * bbmat

        p0 = jax.scipy.stats.multivariate_normal.logpdf(z_try, mean=mu, cov=cov)
        p1 = jax.scipy.stats.multivariate_normal.logpdf(
            z_try, mean=mu, cov=cov_
        )
        w = jnp.exp(p1 - p0)

        # Mann-Whitney. It is assumed z_try was in ascending order.
        n_ = jnp.sum(w)
        u_ = jnp.sum(w * (jnp.arange(self.popsize) + 0.5))

        u_mu = self.popsize * n_ * 0.5
        u_sigma = jnp.sqrt(self.popsize * n_ * (self.popsize + n_ + 1) / 12.0)
        cum = jax.scipy.stats.norm.cdf(u_, loc=u_mu, scale=u_sigma)

        decrease = cum < rho
        eta_out = jax.lax.select(
            decrease,
            (1 - c) * eta_sigma + c * eta_sigma_init,
            jnp.minimum(1, (1 + c) * eta_sigma),
        )
        return eta_out
