import jax
import jax.numpy as jnp
from functools import partial


class Strategy(object):
    def __init__(self, num_dims: int, popsize: int):
        """Base Abstract Class for an Evolutionary Strategy."""
        self.num_dims = num_dims
        self.popsize = popsize

    def default_params(self):
        """Return default parameters of evolutionary strategy."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, rng, params):
        """`initialize` the evolutionary strategy."""
        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng, params)

        # Add best performing parameters/fitness tracker
        state["best_member"] = jnp.zeros(self.num_dims)
        state["best_fitness"] = 20e10
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng, state, params):
        """`ask` for new parameter candidates to evaluate next."""
        # Generate proposal based on strategy-specific ask method
        x, state = self.ask_strategy(rng, state, params)
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, x, fitness, state, params):
        """`tell` performance data for strategy state update."""
        # Update the search state based on strategy-specific update
        state = self.tell_strategy(x, fitness, state, params)

        # Check if there is a new best member
        best_in_gen = jnp.argmin(fitness)
        best_in_gen_fitness, best_in_gen_member = fitness[best_in_gen], x[best_in_gen]
        replace_best = best_in_gen_fitness < state["best_fitness"]
        state["best_fitness"] = jax.lax.select(
            replace_best, best_in_gen_fitness, state["best_fitness"]
        )
        state["best_member"] = jax.lax.select(
            replace_best, best_in_gen_member, state["best_member"]
        )
        return state

    def initialize_strategy(self, rng, params):
        """Search-specific `initialize` method."""
        raise NotImplementedError

    def tell_strategy(self, x, fitness, state, params):
        """Search-specific `tell` update."""
        raise NotImplementedError

    def ask_strategy(self, rng, state, params):
        """Search-specific `ask` request."""
        raise NotImplementedError
