import chex
from typing import Tuple


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

    def independent(
        self, x: chex.Array, fitness: chex.Array
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        # Reshape flat fitness/search vector into subpopulation array then tell
        # batch_fitness -> Shape: (subpops, popsize_per_subpop)
        # batch_x -> Shape: (subpops, popsize_per_subpop, num_dims)
        # Base independent update of each strategy only with subpop-specific data
        batch_fitness = fitness.reshape(self.num_subpops, self.sub_popsize)
        batch_x = x.reshape(self.num_subpops, self.sub_popsize, self.num_dims)
        return batch_fitness, batch_x
