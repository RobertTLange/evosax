import jax
import jax.numpy as jnp
import chex
from typing import Optional, Tuple
from functools import partial
from .. import Strategies
from .protocol import Protocol


class BatchStrategy(object):
    def __init__(
        self,
        strategy_name: str,
        num_dims: int,
        popsize: int,
        num_subpops: int,
        strategy_kwargs: dict = {},
        communication: str = "independent",
        n_devices: Optional[int] = None,
    ):
        """Parallelization/vectorization of ES across subpopulations."""
        self.num_subpops = num_subpops
        self.strategy_name = strategy_name
        self.num_dims = num_dims
        self.popsize = popsize
        self.sub_popsize = int(popsize / num_subpops)
        self.strategy = Strategies[self.strategy_name](
            popsize=self.sub_popsize, num_dims=self.num_dims, **strategy_kwargs
        )
        self.protocol = Protocol(
            communication, self.num_dims, self.num_subpops, self.sub_popsize
        )

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1:
            print(
                "BatchStrategy: More than one device detected. Please make sure"
                " that the number of ES subpopulations divides evenly across"
                " the number of devices to pmap/parallelize over."
            )

    @property
    def default_params(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        base_params = self.strategy.default_params
        # Repeat the default parameters for each subpopulation
        repeated_params = {}
        for k, v in base_params.items():
            repeated_params[k] = jnp.stack(self.num_subpops * [v])
        return repeated_params

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the batch evolution strategy."""
        batch_rng = jax.random.split(rng, self.num_subpops)
        state = jax.vmap(self.strategy.initialize, in_axes=(0, 0))(
            batch_rng, params
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        batch_rng = jax.random.split(rng, self.num_subpops)
        batch_x, state = jax.vmap(self.strategy.ask, in_axes=(0, 0, 0))(
            batch_rng, state, params
        )
        # Flatten subpopulation proposals back into flat vector
        # batch_x -> Shape: (subpops, popsize_per_subpop, num_dims)
        x_re = batch_x.reshape(self.popsize, self.num_dims)
        return x_re, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Communicate and reshape information between subpopulations
        batch_fitness, batch_x = self.protocol.broadcast(x, fitness)
        state = jax.vmap(self.strategy.tell, in_axes=(0, 0, 0, 0))(
            batch_x, batch_fitness, state, params
        )
        return state
