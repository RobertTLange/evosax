import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from ..utils import GradientOptimizer


class ARS(Strategy):
    def __init__(
        self,
        num_dims: int,
        popsize: int,
        elite_ratio: float = 0.1,
        opt_name: str = "sgd",
    ):
        """Augmented Random Search (Mania et al., 2018)
        Reference: https://arxiv.org/pdf/1803.07055.pdf"""
        super().__init__(num_dims, popsize)
        assert not self.popsize & 1, "Population size must be even"
        # ARS performs antithetic sampling & allows you to select
        # "b" elite perturbation directions for the update
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize / 2 * self.elite_ratio)
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "ARS"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        es_params = {
            "sigma_init": 0.03,
            "sigma_decay": 0.999,
            "sigma_limit": 0.01,
            "init_min": 0.0,
            "init_max": 0.0,
        }
        params = {**es_params, **self.optimizer.default_params}
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        es_state = {
            "mean": initialization,
            "sigma": params["sigma_init"],
        }
        state = {**es_state, **self.optimizer.initialize(params)}
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        z_plus = jax.random.normal(
            rng,
            (int(self.popsize / 2), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state["mean"] + state["sigma"] * z
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Reconstruct noise from last mean/std estimates
        noise = (x - state["mean"]) / state["sigma"]
        noise_1 = noise[: int(self.popsize / 2)]
        fit_1 = fitness[: int(self.popsize / 2)]
        fit_2 = fitness[int(self.popsize / 2) :]
        elite_idx = jnp.minimum(fit_1, fit_2).argsort()[: self.elite_popsize]

        fitness_elite = jnp.concatenate([fit_1[elite_idx], fit_2[elite_idx]])
        # Add small constant to ensure non-zero division stability
        sigma_fitness = jnp.std(fitness_elite) + 1e-05
        fit_diff = fit_1[elite_idx] - fit_2[elite_idx]
        fit_diff_noise = jnp.dot(noise_1[elite_idx].T, fit_diff)

        theta_grad = 1.0 / (self.elite_popsize * sigma_fitness) * fit_diff_noise
        # print(jnp.linalg.norm(theta_grad), sigma_fitness)
        # Grad update using optimizer instance - decay lrate if desired
        state = self.optimizer.step(theta_grad, state, params)
        state = self.optimizer.update(state, params)
        # Update lrate and standard deviation based on min and decay
        state["sigma"] *= params["sigma_decay"]
        state["sigma"] = jnp.maximum(state["sigma"], params["sigma_limit"])
        return state
