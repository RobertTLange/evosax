import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
import chex
import flax
from functools import partial
from .batch import BatchStrategy
from ... import Strategies


class MetaStrategy(BatchStrategy):
    def __init__(
        self,
        meta_strategy_name: str,
        inner_strategy_name: str,
        meta_params: List[str],
        num_dims: int,
        popsize: int,
        num_subpops: int,
        meta_strategy_kwargs: dict = {},
        inner_strategy_kwargs: dict = {},
        communication: str = "independent",
        n_devices: Optional[int] = None,
    ):
        # Initialize the batch strategy - subpops of inner strategies
        super().__init__(
            inner_strategy_name,
            num_dims,
            popsize,
            num_subpops,
            inner_strategy_kwargs,
            communication,
            n_devices,
        )
        self.meta_strategy_name = meta_strategy_name
        self.meta_params = meta_params
        self.num_meta_dims = len(self.meta_params)
        self.meta_strategy = Strategies[self.meta_strategy_name](
            popsize=self.num_subpops,
            num_dims=self.num_meta_dims,
            **meta_strategy_kwargs
        )

    @property
    def default_params_meta(self) -> chex.ArrayTree:
        """Return default parameters of meta-evolution strategy."""
        base_params = flax.serialization.to_state_dict(
            self.meta_strategy.default_params
        )
        # Copy over default parameters for init min/init max
        init_val = []
        for k in self.meta_params:
            init_val.append(getattr(self.strategy.default_params, k))
        base_params["init_min"] = jnp.array(init_val)
        base_params["init_max"] = jnp.array(init_val)
        return flax.serialization.from_state_dict(
            self.meta_strategy.default_params, base_params
        )

    @partial(jax.jit, static_argnums=(0,))
    def ask_meta(
        self,
        rng: chex.PRNGKey,
        meta_state: chex.ArrayTree,
        meta_params: chex.ArrayTree,
        inner_params: chex.ArrayTree,
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for meta-parameters of different subpopulations."""
        meta_x, meta_state = self.meta_strategy.ask(
            rng, meta_state, meta_params
        )
        meta_x = meta_x.reshape(-1, self.num_meta_dims)
        re_inner_params = flax.serialization.to_state_dict(inner_params)
        for i, k in enumerate(self.meta_params):
            re_inner_params[k] = meta_x[:, i]
        return (
            flax.serialization.from_state_dict(inner_params, re_inner_params),
            meta_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def initialize_meta(
        self, rng: chex.PRNGKey, meta_params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the meta-evolution strategy."""
        return self.meta_strategy.initialize(rng, meta_params)

    @partial(jax.jit, static_argnums=(0,))
    def tell_meta(
        self,
        inner_params: chex.ArrayTree,
        fitness: chex.Array,
        meta_state: chex.ArrayTree,
        meta_params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for meta-strategy state update."""
        # TODO: Default - mean subpop fitness -> more flexible (min/max/median)
        batch_fitness = fitness.reshape(self.num_subpops, self.sub_popsize)
        meta_fitness = batch_fitness.mean(axis=1)
        # Reconstruct meta_x for dict of inner params
        meta_x = []
        re_inner_params = flax.serialization.to_state_dict(inner_params)
        for i, k in enumerate(self.meta_params):
            meta_x.append(re_inner_params[k].reshape(-1, 1))
        meta_x = jnp.concatenate(meta_x, axis=1)

        # Update the meta strategy
        meta_state = self.meta_strategy.tell(
            meta_x, meta_fitness, meta_state, meta_params
        )
        return meta_state
