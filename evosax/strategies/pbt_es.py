import jax
import jax.numpy as jnp
from ..strategy import Strategy


class PBT_ES(Strategy):
    def __init__(self, popsize: int, num_dims: int):
        """Synchronous version of Population-Based Training."""
        super().__init__(num_dims, popsize)

    @property
    def default_params(self):
        return {
            "noise_scale": 0.1,
            "truncation_selection": 0.2,
            "init_min": -2,  # Param. init range - min
            "init_max": 2,  # Param. init range - max
        }

    def initialize_strategy(self, rng, params):
        """
        `initialize` the differential evolution strategy.
        """
        state = {
            "archive": jax.random.uniform(
                rng,
                (self.popsize, self.num_dims),
                minval=params["init_min"],
                maxval=params["init_max"],
            ),
            "fitness": jnp.zeros(self.popsize) - 20e10,
        }
        return state

    def ask_strategy(self, rng, state, params):
        """
        `ask` for new proposed candidates to evaluate next.
        Perform explore-exploit step.
        1) Check exploit criterion (e.g. in top 20% of performer).
        2) If not exploit: Copy hyperparams from id and explore/perturb around.
        3) Return new hyperparameters and copy_id (same if exploit)
        """
        rng_members = jax.random.split(rng, self.popsize)
        member_ids = jnp.arange(self.popsize)
        exploit_bool, copy_id, hyperparams = jax.vmap(
            single_member_exploit, in_axes=(0, None, None, None)
        )(member_ids, state["archive"], state["fitness"], params)
        hyperparams = jax.vmap(single_member_explore, in_axes=(0, 0, 0, None))(
            rng_members, exploit_bool, hyperparams, params
        )
        return copy_id, hyperparams, state

    def tell_strategy(self, x, fitness, state, params):
        """
        `tell` update to ES state. - Only copy if perfomance has improved.
        """
        replace = fitness >= state["fitness"]
        state["archive"] = (
            jnp.expand_dims(replace, 1) * x
            + (1 - jnp.expand_dims(replace, 1)) * state["archive"]
        )
        state["fitness"] = replace * fitness + (1 - replace) * state["fitness"]
        return state


def single_member_exploit(member_id, archive, fitness, params):
    """Get the top and bottom performers."""
    best_id = jnp.argmax(fitness)
    exploit_bool = member_id != best_id  # Copy if worker not best
    copy_id = jax.lax.select(exploit_bool, best_id, member_id)
    hyperparams_copy = archive[copy_id]
    return exploit_bool, copy_id, hyperparams_copy


def single_member_explore(rng, exploit_bool, hyperparams, params):
    explore_noise = jax.random.normal(rng, hyperparams.shape) * params["noise_scale"]
    hyperparams_explore = jax.lax.select(
        exploit_bool, hyperparams + explore_noise, hyperparams
    )
    return hyperparams_explore


if __name__ == "__main__":
    from functools import partial

    @partial(jax.vmap, in_axes=(0, 0, None))
    def step(theta, h, lrate):
        """Perform GradAscent step on quadratic surrogate objective (maximize!)."""
        surrogate_grad = -2.0 * h * theta
        return theta + lrate * surrogate_grad

    @partial(jax.vmap, in_axes=(0,))
    def evaluate(theta):
        """Ground truth objective (e.g. val loss) as in Jaderberg et al. 2016."""
        return 1.2 - jnp.sum(theta ** 2)

    @partial(jax.vmap, in_axes=(0, 0))
    def surrogate_objective(theta, h):
        """Surrogate objective (with hyperparams h) as in Jaderberg et al. 2016."""
        return 1.2 - jnp.sum(h * theta ** 2)

    rng = jax.random.PRNGKey(1)
    strategy = PBT_ES(2, 2)
    params = strategy.default_params
    params["noise_scale"] = 0.5
    state = strategy.initialize(rng, params)

    # set the state manually for init
    theta = jnp.array([[0.9, 0.9], [0.9, 0.9]])
    h = jnp.array([[0, 1], [1, 0]])

    # Run 10 steps and evaluate final performance
    fitness_log = []
    theta_log = []
    for gen in range(20):
        rng, rng_gen = jax.random.split(rng, 2)
        for i in range(10):
            theta = step(theta, h, 0.01)
            theta_log.append(theta)
        fitness = evaluate(theta)
        state = strategy.tell(h, fitness, state, params)
        copy_id, h, state = strategy.ask(rng_gen, state, params)
        theta = theta[copy_id]
        fitness_log.append(fitness)

    theta_log = jnp.array(theta_log)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(fitness_log)
    axs[1].scatter(theta_log[:, 0], theta_log[:, 1], s=8)
