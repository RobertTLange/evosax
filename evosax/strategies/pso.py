import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from flax import struct


@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    velocity: chex.Array
    best_archive: chex.Array
    best_archive_fitness: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    inertia_coeff: float = 0.75  # w momentum of velocity
    cognitive_coeff: float = 1.5  # c_1 cognitive "force" multiplier
    social_coeff: float = 2.0  # c_2 social "force" multiplier
    init_min: float = -0.1
    init_max: float = 0.1
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class PSO(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Particle Swarm Optimization (Kennedy & Eberhart, 1995)
        Reference: https://ieeexplore.ieee.org/document/488968"""
        super().__init__(num_dims, popsize)
        self.strategy_name = "PSO"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.popsize) + jnp.finfo(jnp.float32).max,
            velocity=jnp.zeros((self.popsize, self.num_dims)),
            best_archive=initialization,
            best_archive_fitness=jnp.zeros(self.popsize)
            + jnp.finfo(jnp.float32).max,
            best_member=initialization.mean(axis=0),
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
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
            state.archive,
            state.velocity,
            state.best_archive,
            state.best_archive_fitness,
            params.inertia_coeff,
            params.cognitive_coeff,
            params.social_coeff,
        )
        # Update particle positions with velocity
        x = state.archive + vel
        return jnp.squeeze(x), state.replace(velocity=vel)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """
        `tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        replace = fitness <= state.best_archive_fitness
        best_archive = (
            jnp.expand_dims(replace, 1) * x
            + (1 - jnp.expand_dims(replace, 1)) * state.best_archive
        )
        best_archive_fitness = (
            replace * fitness + (1 - replace) * state.best_archive_fitness
        )
        return state.replace(
            mean=x.mean(axis=0),
            fitness=fitness,
            archive=x,
            best_archive=best_archive,
            best_archive_fitness=best_archive_fitness,
        )


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
