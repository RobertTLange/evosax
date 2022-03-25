import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class PSO(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Particle Swarm Optimization (Kennedy & Eberhart, 1995)
        Reference: https://ieeexplore.ieee.org/document/488968"""
        super().__init__(num_dims, popsize)
        self.strategy_name = "PSO"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        return {
            "inertia_coeff": 0.75,  # w momentum of velocity
            "cognitive_coeff": 1.5,  # c_1 cognitive "force" multiplier
            "social_coeff": 2.0,  # c_2 social "force" multiplier
            "init_min": -0.1,
            "init_max": 0.1,
        }

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "mean": initialization.mean(axis=0),
            "archive": initialization,
            "fitness": jnp.zeros(self.popsize) + 20e10,
            "velocity": jnp.zeros((self.popsize, self.num_dims)),
        }
        state["best_archive"] = state["archive"][:]
        state["best_archive_fitness"] = state["fitness"][:]
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """
        `ask` for new proposed candidates to evaluate next.
        1. Update v_i(t+1) velocities base on:
          - Inertia: w * v_i(t)
          - Cognitive: c_1 * r_1 * (p_(i, lb)(t) - x_i(t))
          - Social: c_2 * r_2 * (p_(gb)(t) - x_i(t))
        2. Update "particle" positions: x_i(t+1) = x_i(t) + v_i(t+1)
        """
        rng_members = jax.random.split(rng, self.popsize)
        member_ids = jnp.arange(self.popsize)
        vel = jax.vmap(
            single_member_velocity,
            in_axes=(0, 0, None, None, None, None, None, None, None),
        )(
            rng_members,
            member_ids,
            state["archive"],
            state["velocity"],
            state["best_archive"],
            state["best_archive_fitness"],
            params["inertia_coeff"],
            params["cognitive_coeff"],
            params["social_coeff"],
        )
        # Update particle positions with velocity
        y = state["archive"] + vel
        state["velocity"] = vel
        return jnp.squeeze(y), state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """
        `tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        state["fitness"] = fitness
        state["archive"] = x
        replace = fitness <= state["best_archive_fitness"]
        state["best_archive"] = (
            jnp.expand_dims(replace, 1) * x
            + (1 - jnp.expand_dims(replace, 1)) * state["best_archive"]
        )
        state["best_archive_fitness"] = (
            replace * fitness + (1 - replace) * state["best_archive_fitness"]
        )

        # Keep mean across stored archive around for evaluation protocol
        state["mean"] = state["archive"].mean(axis=0)
        return state


def single_member_velocity(
    rng: chex.PRNGKey,
    member_id: int,
    archive: chex.Array,
    velocity: chex.Array,
    best_archive: chex.Array,
    best_fitness: chex.Array,
    inertia_coeff: float,
    cognitive_coeff: float,
    social_coeff: float,
):
    """Update v_i(t+1) velocities based on: Inertia, Cognitive, Social."""
    # Sampling one shared r1, r2 across dims of one member seems more robust!
    # r1, r2 = jax.random.uniform(rng, (2, archive.shape[1]))
    r1, r2 = jax.random.uniform(rng, (2,))
    global_best_id = jnp.argmin(best_fitness)
    global_best = best_archive[global_best_id]
    vel_new = (
        inertia_coeff * velocity[member_id]
        + cognitive_coeff * r1 * (best_archive[member_id] - archive[member_id])
        + social_coeff * r2 * (global_best - archive[member_id])
    )
    return vel_new
