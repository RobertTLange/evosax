import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class DE(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Differential Evolution (Storn & Price, 1997)
        Reference: https://tinyurl.com/4pje5a74"""
        assert popsize > 6
        super().__init__(num_dims, popsize)
        self.strategy_name = "DE"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        return {
            "mutate_best_vector": True,  # 0 - 'random'
            "num_diff_vectors": 1,  # [1, 2]
            "cross_over_rate": 0.9,  # cross-over probability [0, 1]
            "diff_w": 0.8,  # differential weight (F) [0, 2]
            "init_min": -0.1,
            "init_max": 0.1,
        }

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """
        `initialize` the differential evolution strategy.
        Initialize all population members by randomly sampling
        positions in search-space (defined in `params`).
        """
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
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """
        `ask` for new proposed candidates to evaluate next.
        For each population member x:
        - Pick 3 unique members (a, b, c) from rest of pop. at random.
        - Pick a random dimension R from the parameter space.
        - Compute the member's potentially new position:
            - For each i ∈ {1,...,d}, pick a r_i ~ U(0, 1)
            - If r_i < cr or i = R, set y_i = a_i + F * (b_i - c_i)
            - Else y_i = x_i
        Return new potential position y.
        """
        rng_members = jax.random.split(rng, self.popsize)
        member_ids = jnp.arange(self.popsize)
        y = jax.vmap(single_member_ask, in_axes=(0, 0, None, None, None, None))(
            rng_members,
            member_ids,
            self.num_dims,
            state["archive"],
            state["best_member"],
            params,
        )
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
        replace = fitness <= state["fitness"]
        state["archive"] = (
            jnp.expand_dims(replace, 1) * x
            + (1 - jnp.expand_dims(replace, 1)) * state["archive"]
        )
        state["fitness"] = replace * fitness + (1 - replace) * state["fitness"]
        # Keep mean across stored archive around for evaluation protocol
        state["mean"] = state["archive"].mean(axis=0)
        return state


def single_member_ask(
    rng: chex.PRNGKey,
    member_id: int,
    num_dims: int,
    archive: chex.Array,
    best_member: chex.Array,
    params: chex.ArrayTree,
) -> chex.Array:
    """Perform `ask` steps for single member."""
    x = archive[member_id]

    # Sample a, b and c parameter vectors from rest of population
    rng, rng_vectors, rng_R = jax.random.split(rng, 3)
    # A bit of an awkward hack - sample one additional member to avoid
    # using same vector as x - check condition and select extra if needed
    # Also always sample 6 members - for case where we want two diff vectors
    row_ids = jax.random.choice(
        rng, jnp.arange(archive.shape[0]), (6,), replace=False
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
    a = jax.lax.select(params["mutate_best_vector"], best_member, a)

    # Sample random dimension that will be alter for sure
    R = jax.random.randint(rng_R, (1,), minval=0, maxval=num_dims)

    rng_dims = jax.random.split(rng, num_dims)
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
        rng_dims,
        dim_ids,
        x,
        a,
        b,
        c,
        d,
        e,
        R,
        params["cross_over_rate"],
        params["diff_w"],
        params["num_diff_vectors"] == 2,
    )
    return y


def single_dimension_ask(
    rng: chex.PRNGKey,
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
    r_i = jax.random.uniform(rng, (1,))
    mutate_bool = jnp.logical_or(r_i < cr, dim_id == R)
    y_i = (
        mutate_bool * a[dim_id]
        + diff_w * (b[dim_id] - c[dim_id])
        + diff_w * (d[dim_id] - e[dim_id]) * use_second_diff
        + (1 - mutate_bool) * x[dim_id]
    )
    return y_i
