import jax
import jax.numpy as jnp
from functools import partial
from .strategy import Strategy


class Differential_ES(Strategy):
    def __init__(self, popsize: int, num_dims: int):
        super().__init__(num_dims, popsize)

    @property
    def default_params(self):
        return {
            "crossover_rate": 0.9,  # cross-over probability [0, 1]
            "diff_w": 0.8,  # differential weight (F) [0, 2]
            "init_min": -2,  # Param. init range - min
            "init_max": 2,  # Param. init range - min
        }

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, rng, params):
        """
        `initialize` the differential evolution strategy.
        Initialize all population members by randomly sampling
        positions in search-space (defined in `params`).
        """
        state = {
            "archive": jax.random.uniform(
                rng,
                (self.popsize, self.num_dims),
                minval=params["init_min"],
                maxval=params["init_max"],
            ),
            "fitness": jnp.zeros(self.popsize) + 20e10,
        }
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng, state, params):
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
        y = jax.vmap(single_member_ask, in_axes=(0, 0, None, None, None))(
            rng_members, member_ids, self.num_dims, state["archive"], params
        )
        return jnp.squeeze(y), state

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, x, fitness, state, params):
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
        return state


def single_member_ask(rng, member_id, num_dims, archive, params):
    """Perform `ask` steps for single member."""
    x = archive[member_id]

    # Sample a, b and c parameter vectors from rest of population
    rng, rng_vectors, rng_R = jax.random.split(rng, 3)
    # A bit of an awkward hack - sample one additional member to avoid
    # using same vector as x - check condition and select extra if needed
    row_ids = jax.random.choice(rng, jnp.arange(archive.shape[0]), (4,), replace=False)
    a = jax.lax.select(
        row_ids[0] == member_id, archive[row_ids[3]], archive[row_ids[0]]
    )
    b = jax.lax.select(
        row_ids[1] == member_id, archive[row_ids[3]], archive[row_ids[1]]
    )
    c = jax.lax.select(
        row_ids[2] == member_id, archive[row_ids[3]], archive[row_ids[2]]
    )

    # Sample random dimension that will be alter for sure
    R = jax.random.randint(rng_R, (1,), minval=0, maxval=num_dims)

    rng_dims = jax.random.split(rng, num_dims)
    dim_ids = jnp.arange(num_dims)
    y = jax.vmap(
        single_dimension_ask, in_axes=(0, 0, None, None, None, None, None, None, None)
    )(rng_dims, dim_ids, x, a, b, c, R, params["crossover_rate"], params["diff_w"])
    return y


def single_dimension_ask(rng, dim_id, x, a, b, c, R, cr, diff_w):
    """Perform `ask` step for single dimension."""
    r_i = jax.random.uniform(rng, (1,))
    mutate_bool = jnp.logical_or(r_i < cr, dim_id == R)
    y_i = (
        mutate_bool * a[dim_id]
        + diff_w * (b[dim_id] - c[dim_id])
        + (1 - mutate_bool) * x[dim_id]
    )
    return y_i
