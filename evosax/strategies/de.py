import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness_archive: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class EvoParams:
    mutate_best_vector: bool = True  # False - 'random'
    num_diff_vectors: int = 1  # [1, 2]
    cross_over_rate: float = 0.9  # cross-over probability [0, 1]
    diff_w: float = 0.8  # differential weight (F) [0, 2]
    init_min: float = -0.1
    init_max: float = 0.1
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class DE(Strategy):
    def __init__(
        self,
        population_size: int,
        pholder_params: chex.ArrayTree | chex.Array | None = None,
        **fitness_kwargs: bool | int | float,
    ):
        """Differential Evolution (Storn & Price, 1997)
        Reference: https://tinyurl.com/4pje5a74
        """
        assert population_size > 6, "DE requires population_size > 6."
        super().__init__(population_size, pholder_params, **fitness_kwargs)
        self.strategy_name = "DE"

    @property
    def params_strategy(self) -> EvoParams:
        return EvoParams()

    def init_strategy(self, key: jax.Array, params: EvoParams) -> EvoState:
        """`init` the differential evolution strategy.
        Initialize all population members by randomly sampling
        positions in search-space (defined in `params`).
        """
        initialization = jax.random.uniform(
            key,
            (self.population_size, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness_archive=jnp.zeros(self.population_size)
            + 20e10,  # TODO: what is 20e10?
            best_member=initialization.mean(axis=0),
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: EvoState, params: EvoParams
    ) -> tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next.
        For each population member x:
        - Pick 3 unique members (a, b, c) from rest of pop. at random.
        - Pick a random dimension R from the parameter space.
        - Compute the member's potentially new position:
            - For each i ∈ {1,...,d}, pick a r_i ~ U(0, 1)
            - If r_i < cr or i = R, set y_i = a_i + F * (b_i - c_i)
            - Else y_i = x_i
        Return new potential position y.
        """
        keys = jax.random.split(key, self.population_size)
        member_ids = jnp.arange(self.population_size)
        x = jax.vmap(single_member_ask, in_axes=(0, 0, None, None, None, None))(
            keys,
            member_ids,
            self.num_dims,
            state.archive,
            state.best_member,
            params,
        )
        return jnp.squeeze(x, axis=2), state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        # Replace member in archive if performance was improved
        replace = fitness <= state.fitness_archive
        archive = (
            jnp.expand_dims(replace, 1) * x
            + (1 - jnp.expand_dims(replace, 1)) * state.archive
        )
        fitness_archive = replace * fitness + (1 - replace) * state.fitness_archive
        # Keep mean across stored archive around for evaluation protocol
        mean = archive.mean(axis=0)
        return state.replace(
            mean=mean, archive=archive, fitness_archive=fitness_archive
        )


def single_member_ask(
    key: jax.Array,
    member_id: int,
    num_dims: int,
    archive: chex.Array,
    best_member: chex.Array,
    params: EvoParams,
) -> chex.Array:
    """Perform `ask` steps for single member."""
    x = archive[member_id]

    # Sample a, b and c parameter vectors from rest of population
    key, key_row_ids, key_R = jax.random.split(key, 3)
    # A bit of an awkward hack - sample one additional member to avoid
    # using same vector as x - check condition and select extra if needed
    # Also always sample 6 members - for case where we want two diff vectors
    row_ids = jax.random.choice(
        key_row_ids, jnp.arange(archive.shape[0]), (6,), replace=False
    )
    a = jax.lax.select(
        row_ids[0] == member_id, archive[row_ids[5]], archive[row_ids[0]]
    )
    b = jax.lax.select(
        row_ids[1] == member_id, archive[row_ids[5]], archive[row_ids[1]]
    )
    c = jax.lax.select(
        row_ids[2] == member_id, archive[row_ids[5]], archive[row_ids[2]]
    )
    d = jax.lax.select(
        row_ids[3] == member_id, archive[row_ids[5]], archive[row_ids[3]]
    )
    e = jax.lax.select(
        row_ids[4] == member_id, archive[row_ids[5]], archive[row_ids[4]]
    )

    # Use best vector instead of random `a` vector if `mutate_vector` == "best"
    a = jax.lax.select(params.mutate_best_vector, best_member, a)

    # Sample random dimension that will be alter for sure
    R = jax.random.randint(key_R, (1,), minval=0, maxval=num_dims)

    keys = jax.random.split(key, num_dims)
    dim_ids = jnp.arange(num_dims)
    y = jax.vmap(
        single_dimension_ask,
        in_axes=(
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )(
        keys,
        dim_ids,
        x,
        a,
        b,
        c,
        d,
        e,
        R,
        params.cross_over_rate,
        params.diff_w,
        params.num_diff_vectors == 2,
    )
    return y


def single_dimension_ask(
    key: jax.Array,
    dim_id: int,
    x: chex.Array,
    a: chex.Array,
    b: chex.Array,
    c: chex.Array,
    d: chex.Array,
    e: chex.Array,
    R: int,
    cr: float,
    diff_w: float,
    use_second_diff: bool,
) -> chex.Array:
    """Perform `ask` step for single dimension."""
    r_i = jax.random.uniform(key, (1,))
    mutate_bool = jnp.logical_or(r_i < cr, dim_id == R)
    y_i = (
        mutate_bool * a[dim_id]
        + diff_w * (b[dim_id] - c[dim_id])
        # Only add second difference vector if desired!
        + diff_w * (d[dim_id] - e[dim_id]) * use_second_diff
        # Mutation - exchange x dim with a dim.
        + (1 - mutate_bool) * x[dim_id]
    )
    return y_i
