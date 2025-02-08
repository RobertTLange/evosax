import chex
import jax
import jax.numpy as jnp
from flax.core import unfreeze
from flax.traverse_util import flatten_dict

from ...utils.helpers import get_ravel_fn
from .decoder import Decoder
from .hyper_networks import HyperNetworkMLP


class HyperDecoder(Decoder):
    def __init__(
        self,
        placeholder_params: chex.ArrayTree | chex.Array,
        key: jax.Array = jax.random.key(0),
        hypernet_config: dict = {
            "num_latent_units": 3,  # Latent units per module kernel/bias
            "num_hidden_units": 2,  # Hidden dimensionality of a_i^j embedding
        },
    ):
        # Get layer shapes of raw network
        flat_params = {
            "/".join(k): v
            for k, v in flatten_dict(unfreeze(placeholder_params)).items()
        }
        network_shapes = jax.tree.map(jnp.shape, flat_params)

        # Instantiate hypernetwork and corresponding parameter reshaper
        self.hyper_network = HyperNetworkMLP(
            **hypernet_config, raw_network_shapes=network_shapes
        )

        net_params = self.hyper_network.init(key)

        self.ravel_params, self.unravel_params = get_ravel_fn(net_params)
        flat_params = self.ravel_params(net_params)
        total_params = flat_params.size

        super().__init__(
            total_params,
            placeholder_params,
        )

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for hypernetwork case."""
        # 0. Reshape genome into params for hypernetwork
        x_params = jax.vmap(self.unravel_params)(x)
        # 1. Project parameters to raw dimensionality using hypernetwork
        hyper_x = jax.jit(jax.vmap(self.hyper_network.apply))(x_params)
        return hyper_x

    def reshape_single(self, x: chex.Array) -> chex.ArrayTree:
        """Reshape a single flat vector using hypernetwork."""
        # 0. Reshape genome into params for hypernetwork
        x_params = self.unravel_params(x)
        # 1. Project parameters to raw dimensionality using hypernetwork
        hyper_x = jax.jit(self.hyper_network.apply)(x_params)
        return hyper_x

    def flatten(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters into flat array."""
        return jax.vmap(self.ravel_params)(x)

    def flatten_single(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters into flat array."""
        return self.ravel_params(x)
