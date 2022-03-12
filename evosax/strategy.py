import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from functools import partial


class Strategy(object):
    def __init__(self, num_dims: int, popsize: int):
        """Base Class for an Evolution Strategy."""
        self.num_dims = num_dims
        self.popsize = popsize

    @property
    def default_params(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        params = self.params_strategy
        # Add shared parameter clipping and archive init params
        params["clip_min"] = -jnp.finfo(jnp.float32).max
        params["clip_max"] = jnp.finfo(jnp.float32).max
        return params

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng, params)

        # Add best performing parameters/fitness tracker/generation counter
        state["best_member"] = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        # Set best fitness to large value - ES minimizes by default!
        state["best_fitness"] = jnp.finfo(jnp.float32).max
        state["gen_counter"] = 0
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        # Generate proposal based on strategy-specific ask method
        x, state = self.ask_strategy(rng, state, params)
        # Clip proposal candidates into allowed range
        x_clipped = jnp.clip(
            jnp.squeeze(x), params["clip_min"], params["clip_max"]
        )
        return x_clipped, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Update the search state based on strategy-specific update
        state = self.tell_strategy(x, fitness, state, params)

        # Update the generation counter
        state["gen_counter"] += 1

        # Check if there is a new best member & update trackers
        best_in_gen = jnp.argmin(fitness)
        best_in_gen_fitness, best_in_gen_member = (
            fitness[best_in_gen],
            x[best_in_gen],
        )
        replace_best = best_in_gen_fitness < state["best_fitness"]
        state["best_fitness"] = jax.lax.select(
            replace_best, best_in_gen_fitness, state["best_fitness"]
        )
        state["best_member"] = jax.lax.select(
            replace_best, best_in_gen_member, state["best_member"]
        )
        return state

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Search-specific `initialize` method. Returns initial state."""
        raise NotImplementedError

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Search-specific `ask` request. Returns proposals & updated state."""
        raise NotImplementedError

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """Search-specific `tell` update. Returns updated state."""
        raise NotImplementedError
