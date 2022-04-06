import jax
import chex
from typing import Tuple
from functools import partial


class Protocol(object):
    def __init__(
        self,
        communication: str,
        num_dims: int,
        num_subpops: int,
        sub_popsize: int,
    ):
        """Base communication protocol for info exchange between subpops."""
        self.communication = communication
        self.num_dims = num_dims
        self.num_subpops = num_subpops
        self.sub_popsize = sub_popsize

        if self.communication == "independent":
            self.broadcast = self.independent
        else:
            raise ValueError(
                "Only implemented independent subpopulations for now."
            )

    @partial(jax.jit, static_argnums=(0,))
    def independent(
        self, batch_x: chex.Array, batch_fitness: chex.Array
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Simply return non-altered candidates & fitness."""
        return batch_x, batch_fitness
