import jax
import jax.numpy as jnp
import chex
from typing import Union, List, Optional
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


class ParameterReshaper(object):
    def __init__(
        self,
        placeholder_params: Union[chex.ArrayTree, chex.Array],
        identity: bool = False,
        n_devices: Optional[int] = None,
    ):
        """Reshape flat parameters vectors into generation eval shape."""
        # Get network shape to reshape
        self.placeholder_params = placeholder_params
        if type(placeholder_params) == dict:
            flat_params = {
                "/".join(k): v
                for k, v in flatten_dict(self.placeholder_params).items()
            }
            self.unflat_shape = jax.tree_map(jnp.shape, self.placeholder_params)
            self.network_shape = jax.tree_map(jnp.shape, flat_params)
            self.total_params = get_total_params(self.network_shape)
            self.l_id = get_layer_ids(self.network_shape)
        elif type(placeholder_params) == FrozenDict:
            self.placeholder_params = unfreeze(self.placeholder_params)
            flat_params = {
                "/".join(k): v
                for k, v in flatten_dict(self.placeholder_params).items()
            }
            self.unflat_shape = jax.tree_map(jnp.shape, self.placeholder_params)
            self.network_shape = jax.tree_map(jnp.shape, flat_params)
            self.total_params = get_total_params(self.network_shape)
            self.l_id = get_layer_ids(self.network_shape)
        else:
            # Classic problem case - no dict but raw array
            self.total_params = self.placeholder_params.shape[0]

        # Special case for no identity mapping (no pytree reshaping)
        if identity:
            self.reshape = jax.jit(self.reshape_identity)
            self.reshape_single = jax.jit(self.reshape_single_flat)
        else:
            self.reshape = jax.jit(self.reshape_network)
            self.reshape_single = jax.jit(self.reshape_single_net)

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1:
            print(
                f"ParameterReshaper: {self.n_devices} devices detected. Please"
                " make sure that the ES population size divides evenly across"
                " the number of devices to pmap/parallelize over."
            )

    def reshape_identity(self, x: chex.Array) -> chex.Array:
        """Return parameters w/o reshaping for evaluation."""
        return x

    def reshape_network(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for a 2D matrix (pop_members, params)."""
        vmap_shape = jax.vmap(self.flat_to_network, in_axes=(0,))
        if self.n_devices > 1:
            x = self.split_params_for_pmap(x)
            map_shape = jax.pmap(vmap_shape)
        else:
            map_shape = vmap_shape
        return map_shape(x)

    def reshape_single_flat(self, x: chex.Array) -> chex.Array:
        """Perform reshaping for a 1D vector (params,)."""
        return x

    def reshape_single_net(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for a 1D vector (params,)."""
        unsqueezed_re = self.flat_to_network(x)
        return unsqueezed_re

    @property
    def vmap_dict(self) -> chex.ArrayTree:
        """Get a dictionary specifying axes to vmap over."""
        vmap_dict = jax.tree_map(lambda x: 0, self.placeholder_params)
        return vmap_dict

    def flat_to_network(self, flat_params: chex.Array) -> chex.ArrayTree:
        """Fill a FrozenDict with new proposed vector of params."""
        new_nn = {}
        layer_keys = self.network_shape.keys()

        # Loop over layers in network
        for i, p_k in enumerate(layer_keys):
            # Select params from flat to vector to be reshaped
            p_flat = jax.lax.dynamic_slice(
                flat_params, (self.l_id[i],), (self.l_id[i + 1] - self.l_id[i],)
            )
            # Reshape parameters into matrix/kernel/etc. shape
            p_reshaped = p_flat.reshape(self.network_shape[p_k])
            # Place reshaped params into dict and increase counter
            new_nn[p_k] = p_reshaped
        return unflatten_dict(
            {tuple(k.split("/")): v for k, v in new_nn.items()}
        )

    def split_params_for_pmap(self, param: chex.Array) -> chex.Array:
        """Helper reshapes param (bs, #params) into (#dev, bs/#dev, #params)."""
        return jnp.stack(jnp.split(param, self.n_devices))


def get_total_params(params: chex.ArrayTree) -> int:
    """Get total number of params in net. Loop over layer modules + params."""
    total_params = 0
    layer_keys = list(params.keys())
    # Loop over layers
    for l_k in layer_keys:
        total_params += jnp.prod(jnp.array(params[l_k]))
    return total_params


def get_layer_ids(network_shape: chex.ArrayTree) -> List[int]:
    """Get indices to target when reshaping single flat net into dict."""
    layer_keys = list(network_shape.keys())
    l_id = [0]
    for l in layer_keys:
        add_pcount = jnp.prod(jnp.array(network_shape[l]))
        l_id.append(int(l_id[-1] + add_pcount))
    return l_id
