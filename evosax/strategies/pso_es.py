import jax
import jax.numpy as jnp
from ..strategy import Strategy


class PSO_ES(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        super().__init__(num_dims, popsize)

    @property
    def params_strategy(self) -> dict:
        return {
            "inertia_coeff": 0.75,  # w momentum of velocity
            "cognitive_coeff": 1.5,  # c_1 cognitive "force" multiplier
            "social_coeff": 2.0,  # c_2 social "force" multiplier
        }

    def initialize_strategy(self, rng, params) -> dict:
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
            "archive": initialization,
            "fitness": jnp.zeros(self.popsize) + 20e10,
            "velocity": jnp.zeros((self.popsize, self.num_dims)),
        }
        state["best_archive"] = state["archive"][:]
        state["best_archive_fitness"] = state["fitness"][:]
        return state

    def ask_strategy(self, rng, state, params):
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

    def tell_strategy(self, x, fitness, state, params):
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
        return state


def single_member_velocity(
    rng,
    member_id,
    archive,
    velocity,
    best_archive,
    best_fitness,
    inertia_coeff,
    cognitive_coeff,
    social_coeff,
):
    """Update v_i(t+1) velocities based on: Inertia, Cognitive, Social."""
    # Sampling one r1, r2 that is shared across dims of one member seems more robust!
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


if __name__ == "__main__":
    from evosax.problems import batch_rosenbrock

    rng = jax.random.PRNGKey(0)
    strategy = PSO_ES(popsize=20, num_dims=2)
    params = strategy.default_params
    state = strategy.initialize(rng, params)
    fitness_log = []

    for i in range(200):
        rng, rng_iter = jax.random.split(rng)
        x, state = strategy.ask(rng_iter, state, params)
        fitness = batch_rosenbrock(x, 1, 100)
        state = strategy.tell(x, fitness, state, params)
        best_id = jnp.argmax(state["best_archive_fitness"])
        fitness_log.append(state["best_archive_fitness"][best_id])
