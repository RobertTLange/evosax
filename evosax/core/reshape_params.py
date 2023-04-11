import jax
import jax.numpy as jnp
import chex
from typing import Union, Optional
from jax import vjp, flatten_util
from jax.tree_util import tree_flatten


def ravel_pytree(pytree):
    leaves, _ = tree_flatten(pytree)
    flat, _ = vjp(ravel_list, *leaves)
    return flat


def ravel_list(*lst):
    return (
        jnp.concatenate([jnp.ravel(elt) for elt in lst])
        if lst
        else jnp.array([])
    )


class ParameterReshaper(object):
    def __init__(
        self,
        placeholder_params: Union[chex.ArrayTree, chex.Array],
        n_devices: Optional[int] = None,
        verbose: bool = True,
    ):
        """Reshape flat parameters vectors into generation eval shape."""
        # Get network shape to reshape
        self.placeholder_params = placeholder_params

        # Set total parameters depending on type of placeholder params
        flat, self.unravel_pytree = flatten_util.ravel_pytree(
            placeholder_params
        )
        self.total_params = flat.shape[0]
        self.reshape_single = jax.jit(self.unravel_pytree)

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1 and verbose:
            print(
                f"ParameterReshaper: {self.n_devices} devices detected. Please"
                " make sure that the ES population size divides evenly across"
                " the number of devices to pmap/parallelize over."
            )

        if verbose:
            print(
                f"ParameterReshaper: {self.total_params} parameters detected"
                " for optimization."
            )

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for a 2D matrix (pop_members, params)."""
        vmap_shape = jax.vmap(self.reshape_single)
        if self.n_devices > 1:
            x = self.split_params_for_pmap(x)
            map_shape = jax.pmap(vmap_shape)
        else:
            map_shape = vmap_shape
        return map_shape(x)

    def multi_reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Reshape parameters lying already on different devices."""
        # No reshaping required!
        vmap_shape = jax.vmap(self.reshape_single)
        return jax.pmap(vmap_shape)(x)

    def flatten(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters (population) into flat array."""
        vmap_flat = jax.vmap(ravel_pytree)
        if self.n_devices > 1:
            # Flattening of pmap paramater trees to apply vmap flattening
            def map_flat(x):
                x_re = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), x)
                return vmap_flat(x_re)

        else:
            map_flat = vmap_flat
        flat = map_flat(x)
        # Out shape: (pop, params)
        return flat

    def flatten_single(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters (single) into flat array."""
        return ravel_pytree(x)

    def multi_flatten(self, x: chex.Array) -> chex.ArrayTree:
        """Flatten parameters lying remaining on different devices."""
        # No reshaping required!
        vmap_flat = jax.vmap(ravel_pytree)
        return jax.pmap(vmap_flat)(x)

    def split_params_for_pmap(self, param: chex.Array) -> chex.Array:
        """Helper reshapes param (bs, #params) into (#dev, bs/#dev, #params)."""
        return jnp.stack(jnp.split(param, self.n_devices))

    @property
    def vmap_dict(self) -> chex.ArrayTree:
        """Get a dictionary specifying axes to vmap over."""
        vmap_dict = jax.tree_map(lambda x: 0, self.placeholder_params)
        return vmap_dict
