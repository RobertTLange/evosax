import jax
import chex
import jax.numpy as jnp
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
        elif self.communication == "best_subpop":
            self.broadcast = self.best_subpop
        else:
            raise ValueError(
                f"{self.communication} is not currently an implemented"
                " protocol."
            )

    @partial(jax.jit, static_argnums=(0,))
    def independent(
        self, batch_x: chex.Array, batch_fitness: chex.Array
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Simply return non-altered candidates & fitness."""
        return batch_x, batch_fitness

    @partial(jax.jit, static_argnums=(0,))
    def best_subpop(
        self, batch_x: chex.Array, batch_fitness: chex.Array
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Find the subpop with the globally best candidate and set all subpops
        to the same as the subpop containing the globally best. Tie currently goes
        to whichever candidate comes first"""
        global_best_arg = batch_fitness.argmin()
        best_subpop_ind = jnp.unravel_index(
            global_best_arg, batch_fitness.shape
        )[0]

        best_subpop_x = batch_x[best_subpop_ind]
        best_subpop_fitness = batch_fitness[best_subpop_ind]

        for i in range(self.num_subpops):
            batch_x = batch_x.at[i].set(best_subpop_x)
            batch_fitness = batch_fitness.at[i].set(best_subpop_fitness)

        return batch_x, batch_fitness
