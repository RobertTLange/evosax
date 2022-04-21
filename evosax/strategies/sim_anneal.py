import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class SimAnneal(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Simulated Annealing (Rasdi Rere et al., 2015)
        Reference: https://www.sciencedirect.com/science/article/pii/S1877050915035759
        """
        super().__init__(num_dims, popsize)
        self.strategy_name = "SimAnneal"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        return {
            "init_min": 0.0,
            "init_max": 0.0,
            "temp_init": 1.0,
            "temp_limit": 0.1,
            "temp_decay": 0.999,
            "boltzmann_const": 5.0,
            "sigma_init": 0.05,
            "sigma_limit": 0.001,
            "sigma_decay": 0.999,
        }

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        rng_init, rng_rep = jax.random.split(rng)
        initialization = jax.random.uniform(
            rng_init,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "mean": initialization,
            "sigma": params["sigma_init"],
            "temp": params["temp_init"],
            "replace_rng": jax.random.uniform(rng_rep, ()),
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new proposed candidates to evaluate next."""
        rng_init, rng_rep = jax.random.split(rng)
        # Sampling of N(0, 1) noise
        z = jax.random.normal(
            rng_init,
            (self.popsize, self.num_dims),
        )
        state["replace_rng"] = jax.random.uniform(rng_rep, ())
        # print(state["best_member"].shape, (state["sigma"] * z).shape)
        x = state["mean"] + state["sigma"] * z
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` update to ES state."""
        best_in_gen = jnp.argmin(fitness)
        gen_fitness, gen_member = fitness[best_in_gen], x[best_in_gen]
        improve_diff = state["best_fitness"] - gen_fitness
        improved = improve_diff > 0

        # Calculate temperature replacement constant (replace by best in gen)
        metropolis = jnp.exp(
            improve_diff / (state["temp"] * params["boltzmann_const"])
        )

        # Replace mean either if improvement or random metropolis acceptance
        rand_replace = jnp.logical_or(
            improved, state["replace_rng"] > metropolis
        )
        # Note: We replace by best member in generation (not completely random)
        state["mean"] = jax.lax.select(rand_replace, gen_member, state["mean"])

        # Update permutation standard deviation
        state["sigma"] = jax.lax.select(
            state["sigma"] > params["sigma_limit"],
            state["sigma"] * params["sigma_decay"],
            state["sigma"],
        )

        state["temp"] = jax.lax.select(
            state["temp"] > params["temp_limit"],
            state["temp"] * params["temp_decay"],
            state["temp"],
        )
        return state
