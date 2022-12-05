import jax
import chex
from typing import Union, Optional


class Decoder(object):
    def __init__(
        self,
        num_encoding_dims: int,
        placeholder_params: Union[chex.ArrayTree, chex.Array],
        n_devices: Optional[int] = None,
    ):
        self.num_encoding_dims = num_encoding_dims
        self.total_params = num_encoding_dims
        self.placeholder_params = placeholder_params
        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1:
            print(
                f"Decoder: {self.n_devices} devices detected. Please"
                " make sure that the ES population size divides evenly across"
                " the number of devices to pmap/parallelize over."
            )

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        raise NotImplementedError

    def reshape_single(self, x: chex.Array) -> chex.ArrayTree:
        raise NotImplementedError
