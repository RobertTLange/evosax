import jax
import jax.numpy as jnp
from functools import partial
from .strategy import Strategy


class Simple_GA(Strategy):
    def __init__(self,
                 popsize: int,
                 num_dims: int,
                 elite_ratio: float):
        super().__init__(num_dims, popsize)
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

    @property
    def default_params(self):
        return {
            "cross_over_rate": 0.5,   # cross-over probability
            "sigma_init": 0.1,        # initial standard deviation
            "sigma_decay": 0.999,     # anneal standard deviation
            "sigma_limit": 0.01,      # stop annealing if less than this
            "elite_ratio": 0.1,       # percentage of the elites
            "forget_best": False,     # forget the historical best elites
            "init_min": -2,          # Param. init range - min
            "init_max": 2            # Param. init range - min
          }

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, rng, params):
        """
        `initialize` the differential evolution strategy.
        Initialize all population members by randomly sampling
        positions in search-space (defined in `params`).
        """
        state = {"archive": jax.random.uniform(
                              rng,
                              (self.elite_popsize, self.num_dims),
                              minval=params["init_min"],
                              maxval=params["init_max"]),
                 "fitness": jnp.zeros(self.elite_popsize) - 20e10,
                 "sigma": params["sigma_init"]}
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng, state, params):
        """
        `ask` for new proposed candidates to evaluate next.
        1. For each member of elite:
          - Sample two current elite members (a & b)
          - Cross over all dims of a with corresponding one from b
            if random number > co-rate
          - Additionally add noise on top of all elite parameters
        """
        rng, rng_eps = jax.random.split(rng)
        rng, rng_idx_a = jax.random.split(rng)
        rng, rng_idx_b = jax.random.split(rng)
        rng_mate = jax.random.split(rng, self.popsize)
        epsilon = jax.random.normal(rng_eps, (self.popsize,
                                              self.num_dims)) * state["sigma"]
        elite_ids = jnp.arange(self.elite_popsize)
        idx_a = jax.random.choice(rng_idx_a, elite_ids, (self.popsize,))
        idx_b = jax.random.choice(rng_idx_b, elite_ids, (self.popsize,))
        members_a = state["archive"][idx_a]
        members_b = state["archive"][idx_b]
        y = jax.vmap(single_mate,
                     in_axes=(0, 0, 0, None))(
                     rng_mate, members_a, members_b,
                     params["cross_over_rate"])
        y += epsilon
        return jnp.squeeze(y), state

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, x, fitness, state, params):
        """
        `tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        # Combine current elite and recent generation info
        fitness = jnp.concatenate([fitness, state["fitness"]])
        solution = jnp.concatenate([x, state["archive"]])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[::-1][0:self.elite_popsize]
        state["fitness"] = fitness[idx]
        state["archive"] = solution[idx]
        # Update mutation epsilon - multiplicative decay
        state["sigma"] = jax.lax.select(state["sigma"] > params["sigma_limit"],
                                        state["sigma"]*params["sigma_limit"],
                                        state["sigma"])
        return state


def single_mate(rng, a, b, cross_over_rate):
    """Only cross-over dims for x% of all dims."""
    idx = jax.random.uniform(rng, (a.shape[0], )) > cross_over_rate
    cross_over_candidate = a * (1 - idx) + b * idx
    return cross_over_candidate


if __name__ == "__main__":
    from evosax.problems import batch_rosenbrock
    rng = jax.random.PRNGKey(0)
    strategy = Simple_GA(popsize=20, num_dims=2, elite_ratio=0.5)
    params = strategy.default_params
    state = strategy.initialize(rng, params)
    x, state = strategy.ask(rng, state, params)
    fitness = -1 * batch_rosenbrock(x, 1, 100)
    state = strategy.tell(x, fitness, state, params)
