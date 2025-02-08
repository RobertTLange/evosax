from functools import partial

import chex
import flax
import jax
import jax.numpy as jnp

from ... import Strategies
from .protocol import Protocol


class BatchStrategy:
    def __init__(
        self,
        strategy_name: str,
        num_dims: int,
        population_size: int,
        num_subpops: int,
        strategy_kwargs: dict = {},
        communication: str = "independent",
        n_devices: int | None = None,
    ):
        """Parallelization/vectorization of ES across subpopulations."""
        self.num_subpops = num_subpops
        self.strategy_name = strategy_name
        self.num_dims = num_dims
        self.population_size = population_size
        self.subpopulation_size = int(population_size / num_subpops)
        self.strategy = Strategies[self.strategy_name](
            population_size=self.subpopulation_size,
            num_dims=self.num_dims,
            **strategy_kwargs,
        )
        self.protocol = Protocol(
            communication, self.num_dims, self.num_subpops, self.subpopulation_size
        )

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1:
            print(
                f"BatchStrategy: {self.n_devices} devices detected. Please make"
                " sure that the number of ES subpopulations"
                f" ({self.num_subpops}) divides evenly across the number of"
                " devices to pmap/parallelize over."
            )
            # Set device-mapped core functionality
            self.init = self.init_pmap
            self.ask_map = self.ask_pmap
            self.tell_map = self.tell_pmap
            self.num_subpops_per_device = int(self.num_subpops / self.n_devices)
            self.population_size_per_device = int(self.population_size / self.n_devices)
        else:
            # Set auto-vectorize core functionality
            self.init = self.init_vmap
            self.ask_map = self.ask_vmap
            self.tell_map = self.tell_vmap
            self.num_subpops_per_device = self.num_subpops
            self.population_size_per_device = self.population_size

    @property
    def default_params(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        base_params = flax.serialization.to_state_dict(self.strategy.default_params)
        # Repeat the default parameters for each subpopulation
        repeated_params = {}
        for k, v in base_params.items():
            repeated_params[k] = jnp.stack(self.num_subpops * [v])
        return flax.serialization.from_state_dict(
            self.strategy.default_params, repeated_params
        )

    @partial(jax.jit, static_argnames=("self",))
    def init_vmap(self, key: jax.Array, params: chex.ArrayTree) -> chex.ArrayTree:
        """Auto-vectorized `init` for the batch evolution strategy."""
        keys = jax.random.split(key, self.num_subpops_per_device)
        state = jax.vmap(self.strategy.init, in_axes=(0, 0))(keys, params)
        return state

    def init_pmap(self, key: jax.Array, params: chex.ArrayTree) -> chex.ArrayTree:
        """Device parallel `init` for the batch evolution strategy."""
        # Tile/reshape both key and params!
        keys = jnp.tile(key, (self.n_devices, 1))
        params_pmap = jax.tree.map(
            lambda x: jnp.stack(jnp.split(x, self.n_devices)), params
        )
        state_pmap = jax.pmap(self.init_vmap)(keys, params_pmap)
        # Reshape from (# device, #subpops_per_device, ...) to (#subpops, ...)
        state = jax.tree.map(
            lambda x: jnp.reshape(x, (self.num_subpops, *x.shape[2:])),
            state_pmap,
        )
        return state

    @partial(jax.jit, static_argnames=("self",))
    def ask(
        self, key: jax.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates."""
        x, state = self.ask_map(key, state, params)
        return x, state

    @partial(jax.jit, static_argnames=("self",))
    def ask_vmap(
        self, key: jax.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> tuple[chex.Array, chex.ArrayTree]:
        """Auto-vectorized `ask` for new parameter candidates."""
        keys = jax.random.split(key, self.num_subpops_per_device)
        batch_x, state = jax.vmap(self.strategy.ask, in_axes=(0, 0, 0))(
            keys, state, params
        )
        # Flatten subpopulation proposals back into flat vector
        # batch_x -> Shape: (subpops, population_size_per_subpop, num_dims)
        x_re = batch_x.reshape(self.population_size_per_device, self.num_dims)
        return x_re, state

    def ask_pmap(
        self, key: jax.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> tuple[chex.Array, chex.ArrayTree]:
        """Device parallel `ask` for new parameter candidates."""
        keys = jnp.tile(key, (self.n_devices, 1))
        params_pmap = jax.tree.map(
            lambda x: jnp.stack(jnp.split(x, self.n_devices)), params
        )
        state_pmap = jax.tree.map(
            lambda x: jnp.stack(jnp.split(x, self.n_devices)), state
        )
        batch_x, state_pmap = jax.pmap(self.ask_vmap)(keys, state_pmap, params_pmap)
        # Flatten subpopulation proposals back into flat vector
        # batch_x -> Shape: (subpops, population_size_per_subpop, num_dims)
        x_re = batch_x.reshape(self.population_size, self.num_dims)
        state_re = jax.tree.map(
            lambda x: jnp.reshape(x, (self.num_subpops, *x.shape[2:])),
            state_pmap,
        )
        return x_re, state_re

    @partial(jax.jit, static_argnames=("self",))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Reshape flat fitness/search vector into subpopulation array then tell
        # batch_fitness -> Shape: (subpops, population_size_per_subpop)
        # batch_x -> Shape: (subpops, population_size_per_subpop, num_dims)
        # Base independent update of each strategy only with subpop-specific data
        batch_fitness = fitness.reshape(self.num_subpops, self.subpopulation_size)
        batch_x = x.reshape(self.num_subpops, self.subpopulation_size, self.num_dims)

        # Communicate and reshape information between subpopulations
        b_x_comm, b_fitness_comm = self.protocol.broadcast(batch_x, batch_fitness)

        # Update the strategy (vectorize vs device parallel)
        state = self.tell_map(b_x_comm, b_fitness_comm, state, params)
        return state

    @partial(jax.jit, static_argnames=("self",))
    def tell_vmap(
        self,
        batch_x: chex.Array,
        batch_fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """Auto-vectorized `tell` performance data for strategy state update."""
        state = jax.vmap(self.strategy.tell, in_axes=(0, 0, 0, 0))(
            batch_x, batch_fitness, state, params
        )
        return state

    def tell_pmap(
        self,
        batch_x: chex.Array,
        batch_fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """Device parallel `tell` performance data for strategy state update."""
        params_pmap = jax.tree.map(
            lambda x: jnp.stack(jnp.split(x, self.n_devices)), params
        )
        state_pmap = jax.tree.map(
            lambda x: jnp.stack(jnp.split(x, self.n_devices)), state
        )
        batch_x_pmap = jnp.stack(jnp.split(batch_x, self.n_devices))
        batch_fitness_pmap = jnp.stack(jnp.split(batch_fitness, self.n_devices))
        state_pmap = jax.pmap(self.tell_vmap)(
            batch_x_pmap, batch_fitness_pmap, state_pmap, params_pmap
        )
        state_re = jax.tree.map(
            lambda x: jnp.reshape(x, (self.num_subpops, *x.shape[2:])),
            state_pmap,
        )
        return state_re
