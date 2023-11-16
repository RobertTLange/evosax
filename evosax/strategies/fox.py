from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from evosax import problems
from ..strategy import Strategy


@struct.dataclass
class EvoState:
    archive: chex.Array
    fitness: chex.Array
    best_archive: chex.Array
    best_archive_fitness: chex.Array
    best_member: chex.Array
    mint: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    init_min: float = -0.1
    init_max: float = 0.1
    c1: float = 0.18
    c2: float = 0.82
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class FOX(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """FOX: a FOX-inspired optimization algorithm (Mohammed & Rashid, 2022)
        Reference: https://link.springer.com/article/10.1007/s10489-022-03533-0"""
        super().__init__(
            popsize, num_dims, pholder_params, n_devices=n_devices, **fitness_kwargs
        )
        self.strategy_name = "FOX"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            archive=initialization,
            fitness=jnp.zeros(self.popsize) + jnp.finfo(jnp.float32).max,
            best_archive=initialization,
            mint=jnp.array([999999]),
            best_archive_fitness=jnp.zeros(self.popsize) + jnp.finfo(jnp.float32).max,
            best_member=initialization.mean(axis=0),
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """
        `ask` for new proposed candidates to evaluate next.
        """
        rng_members = jax.random.split(rng, self.popsize)
        member_ids = jnp.arange(self.popsize)
        x, tt = jax.vmap(
            update_fox_position,
            in_axes=(0, 0, None, None, None, None, None, None, None),
        )(
            rng_members,
            member_ids,
            state.gen_counter,
            params.c1,
            params.c2,
            state.mint,
            state.archive,
            state.best_archive,
            state.best_archive_fitness,
        )
        # TODO: Enhance
        # due to vmap, I get many mints, but it's only one param
        return x, state.replace(mint=tt.mean())

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
        # Replace members if better fitness correspondingly
        # remember default is minimize fitness
        replace = fitness <= state.best_archive_fitness
        best_archive = (
            jnp.expand_dims(replace, 1) * x
            + (1 - jnp.expand_dims(replace, 1)) * state.best_archive
        )
        best_archive_fitness = (
            replace * fitness + (1 - replace) * state.best_archive_fitness
        )
        return state.replace(
            archive=x,
            fitness=fitness,
            best_archive=best_archive,
            best_archive_fitness=best_archive_fitness,
        )


def update_fox_position(
    rng: chex.PRNGKey,
    member_id: int,
    gen_counter: int,
    c1: int,
    c2: int,
    mint: int,
    archive: chex.Array,
    best_archive: chex.Array,
    best_fitness: chex.Array,
):
    """Update position based on: Red Fox Hunting Behavior: Exploration and Exploitation"""
    current_global_best_id = jnp.argmin(best_fitness)
    current_global_best = best_archive[current_global_best_id]

    aa = jnp.array([2.0 * (1.0 - (1.0 / gen_counter))])
    r1 = jax.random.uniform(rng, (1,))
    r2 = jax.random.uniform(rng + 1, (1,))

    p = jnp.array([0.5])

    t1 = jax.random.uniform(rng + 2, (len(current_global_best),))
    sps = current_global_best / t1
    dis = 0.5 * sps * t1
    tt = jnp.mean(t1)
    t = tt / 2.0
    jump = 0.5 * 9.81 * t**2

    new_archive = jnp.where(
        r1 > p,
        jnp.where(r2 > c1, dis * jump * c1, dis * jump * c2),
        (
            current_global_best
            + jax.random.normal(rng + 3, shape=(len(current_global_best),)) * mint * aa
        ),
    )

    new_tt = jnp.where(r1 > p, jnp.where(mint > tt, tt, mint), mint)
    return new_archive.squeeze(), new_tt.squeeze()
