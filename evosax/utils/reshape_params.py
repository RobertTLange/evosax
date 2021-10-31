import jax
import jax.numpy as jnp
from flax.core import FrozenDict


class ParameterReshaper(object):
    def __init__(self, placeholder_params):
        """Reshape flat parameters vectors into generation eval shape."""
        self.placeholder_params = placeholder_params
        self.total_params = get_total_params(self.placeholder_params)
        self.network_shape = get_network_shapes(self.placeholder_params)

    def reshape(self, x):
        """Perform reshaping for a 2D matrix (pop_members, params)."""
        vmap_shape = jax.vmap(flat_to_network, in_axes=(0, None))
        return vmap_shape(x, self.network_shape)

    def reshape_single(self, x):
        """Perform reshaping for a 1D vector (params,)."""
        unsqueezed_re = self.reshape(x.reshape(1, -1))
        squeeze_dict = {}
        layer_keys = list(self.network_shape.keys())
        for l_k in layer_keys:
            place_h_layer = {}
            for p_k in self.network_shape[l_k].keys():
                place_h_layer[p_k] = unsqueezed_re[l_k][p_k].reshape(
                    self.network_shape[l_k][p_k]
                )
            # Add mapping wrapped parameters to dict
            squeeze_dict[l_k] = place_h_layer
        return squeeze_dict

    @property
    def vmap_dict(self):
        """Get a dictionary specifying axes to vmap over."""
        vmap_dict = {}
        layer_keys = list(self.network_shape.keys())
        for l_k in layer_keys:
            place_h_layer = {}
            param_keys = list(self.network_shape[l_k].keys())
            for p_k in param_keys:
                place_h_layer[p_k] = 0
            vmap_dict[l_k] = place_h_layer
        return vmap_dict


def get_total_params(params):
    """Get total number of params in net. Loop over layer modules + params."""
    total_params = 0
    layer_keys = list(params.keys())
    # Loop over layers
    for l_k in layer_keys:
        no_params = get_layer_params(params[l_k])
        # print(l_k, no_params)
        total_params += sum(no_params.values())
    return total_params


def get_layer_params(layer, sum_up=False):
    """Get dict with no params per trafo matrix/vector"""
    param_keys = list(layer.keys())
    counts = {}
    # Loop over params in layer
    for p_k in param_keys:
        counts[p_k] = jnp.prod(jnp.array(layer[p_k].shape))
    return counts


def get_network_shapes(params):
    """Get dict w. shapes per layer/module & list of indexes flat vector."""
    layer_keys = list(params.keys())
    placeh_nn = {}
    for l_k in layer_keys:
        place_h_layer = {}
        param_keys = list(params[l_k].keys())
        for p_k in param_keys:
            place_h_layer[p_k] = params[l_k][p_k].shape
        placeh_nn[l_k] = FrozenDict(place_h_layer)
    return FrozenDict(placeh_nn)


def flat_to_network(flat_params, network_shapes):
    """Fill a FrozenDict with new proposed vector of params."""
    layer_keys = list(network_shapes.keys())
    # print(layer_keys)
    new_nn = {}
    param_counter = 0

    # Loop over layers in network
    for l_k in layer_keys:
        # print(l_k)
        place_h_layer = {}
        # Loop over params in layer
        for p_k in network_shapes[l_k].keys():
            # print(p_k)
            # Select params from flat to vector to be reshaped
            params_to_add = jnp.prod(jnp.array(network_shapes[l_k][p_k]))
            p_flat = flat_params[param_counter : (param_counter + params_to_add)]
            # print(p_flat.shape, network_shapes[l_k][p_k])
            # Reshape parameters into matrix/kernel/etc. shape
            p_reshaped = p_flat.reshape(network_shapes[l_k][p_k])
            # Place reshaped params into dict and increase counter
            place_h_layer[p_k] = p_reshaped
            param_counter += params_to_add

        # Add mapping wrapped parameters to dict
        new_nn[l_k] = place_h_layer
    nn = new_nn
    return nn
