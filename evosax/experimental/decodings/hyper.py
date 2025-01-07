import jax
import jax.numpy as jnp
import chex
from flax.traverse_util import flatten_dict
from flax.core import unfreeze
from typing import Union, Optional
from .decoder import Decoder
from .hyper_networks import HyperNetworkMLP
from ...core import ParameterReshaper, ravel_pytree


class HyperDecoder(Decoder):
    def __init__(
        self,
        placeholder_params: Union[chex.ArrayTree, chex.Array],
        rng: chex.PRNGKey = jax.random.PRNGKey(0),
        hypernet_config: dict = {
            "num_latent_units": 3,  # Latent units per module kernel/bias
            "num_hidden_units": 2,  # Hidden dimensionality of a_i^j embedding
        },
        identity: bool = False,
        n_devices: Optional[int] = None,
    ):
        # Get layer shapes of raw network
        flat_params = {
            "/".join(k): v
            for k, v in flatten_dict(unfreeze(placeholder_params)).items()
        }
        network_shapes = jax.tree_map(jnp.shape, flat_params)

        # Instantiate hypernetwork and corresponding parameter reshaper
        self.hyper_network = HyperNetworkMLP(
            **hypernet_config, raw_network_shapes=network_shapes
        )

        net_params = self.hyper_network.init(rng)
        hyper_reshaper = ParameterReshaper(net_params)

        super().__init__(
            hyper_reshaper.total_params,
            placeholder_params,
            n_devices,
        )
        self.hyper_reshaper = hyper_reshaper
        self.vmap_dict = self.hyper_reshaper.vmap_dict

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for hypernetwork case."""
        # 0. Reshape genome into params for hypernetwork
        x_params = self.hyper_reshaper.reshape(x)
        # 1. Project parameters to raw dimensionality using hypernetwork
        hyper_x = jax.jit(jax.vmap(self.hyper_network.apply))(x_params)
        return hyper_x

    def reshape_single(self, x: chex.Array) -> chex.ArrayTree:
        """Reshape a single flat vector using hypernetwork."""
        # 0. Reshape genome into params for hypernetwork
        x_params = self.hyper_reshaper.reshape_single(x)
        # 1. Project parameters to raw dimensionality using hypernetwork
        hyper_x = jax.jit(self.hyper_network.apply)(x_params)
        return hyper_x

    def flatten(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters into flat array."""
        return jax.vmap(ravel_pytree)(x)

    def flatten_single(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters into flat array."""
        return ravel_pytree(x)
