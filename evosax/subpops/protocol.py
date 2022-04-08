import jax
import chex
import jax.numpy as jnp
from typing import Tuple
from functools import partial
import pdb


class Protocol(object):
    def __init__(
        self,
        communication: str,
        num_dims: int,
        num_subpops: int,
        sub_popsize: int,
        # num_flex_communication: int = 0,
    ):
        """Base communication protocol for info exchange between subpops."""
        self.communication = communication
        self.num_dims = num_dims
        self.num_subpops = num_subpops
        self.sub_popsize = sub_popsize
        # self.subpop_inds = list(range(num_subpops))

        if self.communication == "independent":
            self.broadcast = self.independent
        elif self.communication == "global_best":
            self.broadcast = self.global_best
        elif self.communication == "best_subpop":
            self.broadcast = self.best_subpop
        else:
            raise ValueError(
                f"{self.communication} is not currently an implemented protocol."
            )

    @partial(jax.jit, static_argnums=(0,))
    def independent(
        self, batch_x: chex.Array, batch_fitness: chex.Array
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Simply return non-altered candidates & fitness."""
        return batch_x, batch_fitness


    @partial(jax.jit, static_argnums=(0,))
    def global_best(
        self, batch_x: chex.Array, batch_fitness: chex.Array
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Add the candidate with the globally best fitness to each other subpop
        in place of it's worst fitness candidate.
        Right now assumes that all equal fitnesses will have the same x"""

        # global_best_fitness = batch_fitness.min()
        global_best_arg = batch_fitness.argmin()
        global_best_inds = jnp.unravel_index(global_best_arg, batch_fitness.shape)
        global_best_fitness = batch_fitness[global_best_inds]
        global_best_x = batch_x[global_best_inds]

        # # For jnp.where to be jittable, it needs to have a guaranteed output size
        # subpops_with_best_fitness = jnp.where(batch_fitness == global_best_fitness, size=self.num_subpops * self.sub_popsize, fill_value=-1)[0]


        # # Iterate over all subpops that do not already have the global best fitness
        # for i in (set(self.subpop_inds) - set(subpops_with_best_fitness.tolist())):
        #     batch_x = batch_x.at[i, subpop_worst_arg[i]].set(global_best_x)
        #     batch_fitness = batch_fitness.at[i, subpop_worst_arg[i]].set(global_best_fitness)

        subpops_best_arg = batch_fitness.argmin(axis=1)
        subpop_worst_arg = batch_fitness.argmax(axis=1)
        for i in range(self.num_subpops):

            # Logically needed if to not just overfill all subpops with the same best x
            # Need to figure out jittable way to deal with the "if"
            # if subpops_best_arg[i] == global_best_fitness:
            #     continue
            batch_x = batch_x.at[i, subpop_worst_arg[i]].set(global_best_x)
            batch_fitness = batch_fitness.at[i, subpop_worst_arg[i]].set(global_best_fitness)

        return batch_x, batch_fitness


    @partial(jax.jit, static_argnums=(0,))
    def best_subpop(
        self, batch_x: chex.Array, batch_fitness: chex.Array
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Find the subpop with the globally best candidate and set all subpops
        to the same as the subpop containing the globally best. Tie currently goes
        to whichever candidate comes first"""
        global_best_arg = batch_fitness.argmin()
        best_subpop_ind = jnp.unravel_index(global_best_arg, batch_fitness.shape)[0]

        best_subpop_x = batch_x[best_subpop_ind]
        best_subpop_fitness = batch_fitness[best_subpop_ind]

        for i in range(self.num_subpops):
            batch_x = batch_x.at[i].set(best_subpop_x)
            batch_fitness = batch_fitness.at[i].set(best_subpop_fitness)

        return batch_x, batch_fitness

