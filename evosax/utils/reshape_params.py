import jax.numpy as jnp
from flax.core import FrozenDict


def get_total_params(params):
    """ Get total number of params in net. Loop over layer modules + params. """
    total_params = 0
    layer_keys = list(params.keys())
    # Loop over layers
    for l_k in layer_keys:
        no_params = get_layer_params(params[l_k])
        # print(l_k, no_params)
        total_params += sum(no_params.values())
    return total_params


def get_layer_params(layer, sum_up=False):
    """ Get dict with no params per trafo matrix/vector"""
    param_keys = list(layer.keys())
    counts = {}
    # Loop over params in layer
    for p_k in param_keys:
        counts[p_k] = jnp.prod(jnp.array(layer[p_k].shape))
    return counts


def get_network_shapes(params):
    """ Get dict w. shapes per layer/module & list of indexes flat vector. """
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
    """ Fill a FrozenDict with new proposed vector of params. """
    layer_keys = list(network_shapes.keys())
    new_nn = {}
    param_counter = 0

    # Loop over layers in network
    for l_k in layer_keys:
        place_h_layer = {}
        # Loop over params in layer
        for p_k in network_shapes[l_k].keys():
            params_to_add = jnp.prod(jnp.array(network_shapes[l_k][p_k]))
            p_flat = flat_params[param_counter:
                                 (param_counter + params_to_add)]
            # Reshape parameters into matrix/kernel/etc. shape
            p_reshaped = p_flat.reshape(network_shapes[l_k][p_k])
            # Place reshaped params into dict and increase counter
            place_h_layer[p_k] = p_reshaped
            param_counter += params_to_add

        # Add mapping wrapped parameters to dict
        new_nn[l_k] = FrozenDict(place_h_layer)
    nn = FrozenDict(new_nn)
    return nn
