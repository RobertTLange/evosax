import jax
import jax.numpy as jnp
from functools import partial
import optax
from .strategy import Strategy


class PEPG_ES(Strategy):
    def __init__(
        self,
        num_dims: int,
        popsize: int = 255,
        elite_ratio: float = 0.0,
        learning_rate: float = 0.01,  # Learning rate for std
        learning_rate_decay: float = 0.9999,  # Anneal the lrate
        learning_rate_limit: float = 0.001,
    ):  # Stop annealing lrate
        self.popsize = popsize
        self.num_dims = num_dims
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

        # Use greedy es to select next mu, rather than using drift param
        if self.elite_popsize > 0:
            self.use_elite = True
        else:
            self.use_elite = False
            # Trick for fixed shapes: use entire population as elite
            self.elite_popsize = self.popsize - 1

        # Baseline = average of batch
        assert self.popsize & 1, "Population size must be odd"
        self.batch_size = int((self.popsize - 1) / 2)
        # Setup Adam optimizer with exponential lrate decay schedule
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        # TODO: Open issue on Github
        # lrate_schedule = optax.exponential_decay(
        #     init_value=self.learning_rate,
        #     decay_rate=self.learning_rate_decay,
        #     transition_steps=1,
        #     end_value=self.learning_rate_limit)
        self.optimizer = optax.chain(
            optax.scale_by_adam(eps=1e-4),
            optax.scale(-self.learning_rate)
            # optax.scale_by_schedule(-schedule_fn)
        )

    @property
    def default_params(self):
        return {
            "sigma_init": 0.10,  # initial standard deviation
            "sigma_alpha": 0.20,  # Learning rate for std
            "sigma_decay": 0.999,  # Anneal standard deviation
            "sigma_limit": 0.01,  # Stop annealing if less than this
            "sigma_max_change": 0.2,  # Clip adaptive sigma to 20%
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
            "mu": jax.random.uniform(
                rng,
                (self.num_dims,),
                minval=params["init_min"],
                maxval=params["init_max"],
            ),
            "fitness": jnp.zeros(self.popsize) - 20e10,
            "sigma": jnp.ones(self.num_dims) * params["sigma_init"],
            "epsilon": jnp.zeros((self.batch_size, self.num_dims)),
            "epsilon_full": jnp.zeros((2 * self.batch_size, self.num_dims)),
        }
        state["optimizer_state"] = self.optimizer.init(state["mu"])
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng, state, params):
        """
        `ask` for new proposed candidates to evaluate next.
        """
        rng, rng_eps = jax.random.split(rng)
        # Antithetic sampling - "positive"/"negative noise"
        state["epsilon"] = jax.random.normal(
            rng_eps, (self.batch_size, self.num_dims)
        ) * state["sigma"].reshape(1, self.num_dims)
        # Note: No average baseline shenanigans as in estool version
        state["epsilon_full"] = jnp.concatenate([state["epsilon"], -state["epsilon"]])
        epsilon = jnp.concatenate(
            [jnp.zeros((1, self.num_dims)), state["epsilon_full"]]
        )
        y = state["mu"].reshape(1, self.num_dims) + epsilon
        return jnp.squeeze(y), state

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, x, fitness, state, params):
        """
        `tell` update to ES state.
        """
        b, fitness_offset = fitness[0], 1
        fitness_sub = fitness[fitness_offset:]
        idx = jnp.argsort(fitness_sub)[::-1][: self.elite_popsize]

        # # Keep track of best performers
        # best_fitness = fitness_sub[idx[0]]
        # best_mu = jax.lax.select(best_fitness > b,
        #                          state["mu"] + state["epsilon_full"][idx[0]],
        #                          state["mu"])
        # best_fitness = jax.lax.select(best_fitness > b,
        #                               fitness[idx[0]],
        #                               b)

        # Compute both mu updates (elite/optimizer-base) and select
        mu_elite = state["mu"] + state["epsilon_full"][idx].mean(axis=0)
        rT = fitness_sub[: self.batch_size] - fitness_sub[self.batch_size :]
        change_mu = jnp.dot(rT, state["epsilon"])
        updates, opt_state = self.optimizer.update(change_mu, state["optimizer_state"])
        mu_optim = optax.apply_updates(state["mu"], updates)
        state["optimizer_state"] = opt_state
        state["mu"] = jax.lax.select(self.use_elite, mu_elite, mu_optim)

        # Adaptive sigma - normalization
        stdev_reward = fitness_sub.std()
        S = (
            state["epsilon"] * state["epsilon"]
            - (state["sigma"] * state["sigma"]).reshape(1, self.num_dims)
        ) / state["sigma"].reshape(1, self.num_dims)
        reward_avg = (
            fitness_sub[: self.batch_size] + fitness_sub[self.batch_size :]
        ) / 2.0
        rS = reward_avg - b
        delta_sigma = (jnp.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

        # adjust sigma according to the adaptive sigma calculation
        # for stability, don't let sigma move more than 10% of orig value
        change_sigma = params["sigma_alpha"] * delta_sigma
        change_sigma = jnp.minimum(
            change_sigma, params["sigma_max_change"] * state["sigma"]
        )
        change_sigma = jnp.maximum(
            change_sigma, -params["sigma_max_change"] * state["sigma"]
        )
        state["sigma"] += change_sigma
        # Update via multiplication with [1, ..., sigma_decay, ... 1] array based on booleans
        decay_part = (state["sigma"] > params["sigma_limit"]) * params["sigma_decay"]
        non_decay_part = state["sigma"] <= params["sigma_limit"]
        update_array = decay_part + non_decay_part
        state["sigma"] = update_array * state["sigma"]
        return state


if __name__ == "__main__":
    from evosax.problems import batch_quadratic
    from evosax.utils import FitnessShaper

    rng = jax.random.PRNGKey(0)
    strategy = PEPG_ES(popsize=51, num_dims=3, learning_rate=0.02)
    fit_shaper = FitnessShaper(z_score_fitness=True)

    params = strategy.default_params
    state = strategy.initialize(rng, params)
    fitness_log = []
    num_iters = 100
    for t in range(num_iters):
        rng, rng_iter = jax.random.split(rng)
        x, state = strategy.ask(rng_iter, state, params)
        fitness = batch_quadratic(x)
        fitness_shaped = fit_shaper.apply(x, fitness)

        state = strategy.tell(x, fitness_shaped, state, params)
        best_id = jnp.argmin(fitness)
        print(t, fitness[best_id], state["mu"])
        fitness_log.append(fitness[best_id])
