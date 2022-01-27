import jax
import jax.numpy as jnp
from ..strategy import Strategy
from ..utils import GradientOptimizer


class PEPG_ES(Strategy):
    def __init__(
        self,
        num_dims: int,
        popsize: int,
        elite_ratio: float = 0.1,
        opt_name: str = "sgd",
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
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)

    @property
    def params_strategy(self):
        es_params = {
            "sigma_init": 0.10,  # initial standard deviation
            "sigma_decay": 0.999,  # Anneal standard deviation
            "sigma_limit": 0.01,  # Stop annealing if less than this
            "sigma_alpha": 0.20,  # Learning rate for std
            "sigma_max_change": 0.2,  # Clip adaptive sigma to 20%
        }
        params = {**es_params, **self.optimizer.default_params}
        return params

    def initialize_strategy(self, rng, params):
        """
        `initialize` the differential evolution strategy.
        Initialize all population members by randomly sampling
        positions in search-space (defined in `params`).
        """
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        es_state = {
            "mean": initialization,
            "fitness": jnp.zeros(self.popsize) - 20e10,
            "lrate": params["lrate_init"],
            "sigma": jnp.ones(self.num_dims) * params["sigma_init"],
            "epsilon": jnp.zeros((self.batch_size, self.num_dims)),
            "epsilon_full": jnp.zeros((2 * self.batch_size, self.num_dims)),
        }
        state = {**es_state, **self.optimizer.initialize(params)}
        return state

    def ask_strategy(self, rng, state, params):
        """
        `ask` for new proposed candidates to evaluate next.
        """
        rng, rng_eps = jax.random.split(rng)
        # Antithetic sampling - "positive"/"negative noise"
        state["epsilon"] = jax.random.normal(
            rng_eps, (self.batch_size, self.num_dims)
        ) * state["sigma"].reshape(1, self.num_dims)
        # Note: No average baseline shenanigans as in estool version
        state["epsilon_full"] = jnp.concatenate(
            [state["epsilon"], -state["epsilon"]]
        )
        epsilon = jnp.concatenate(
            [jnp.zeros((1, self.num_dims)), state["epsilon_full"]]
        )
        y = state["mean"].reshape(1, self.num_dims) + epsilon
        return jnp.squeeze(y), state

    def tell_strategy(self, x, fitness, state, params):
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
        mu_elite = state["mean"] + state["epsilon_full"][idx].mean(axis=0)
        rT = fitness_sub[: self.batch_size] - fitness_sub[self.batch_size :]
        theta_grad = jnp.dot(rT, state["epsilon"])
        state = self.optimizer.step(theta_grad, state, params)
        state = self.optimizer.update(state, params)
        state["mean"] = jax.lax.select(self.use_elite, mu_elite, state["mean"])

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
        decay_part = (state["sigma"] > params["sigma_limit"]) * params[
            "sigma_decay"
        ]
        non_decay_part = state["sigma"] <= params["sigma_limit"]
        update_array = decay_part + non_decay_part
        state["sigma"] = update_array * state["sigma"]
        return state
