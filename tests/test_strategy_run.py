import jax
import jax.numpy as jnp
from evosax import Strategies
from evosax.problems import batch_rosenbrock, batch_quadratic
from evosax.utils import FitnessShaper
from functools import partial

num_iters = 25


"""
OpenNES: params["lrate"] = 0.015, num_iters = 200
PEPG: params["lrate"] = 0.01, num_iters = 500
"""


def test_strategy_run(strategy_name):
    # Loop over all strategies and test ask API
    rng = jax.random.PRNGKey(0)
    Strat = Strategies[strategy_name]
    # PBT also returns copy ID integer - treat separately
    if strategy_name in ["Persistent_ES", "Open_NES"]:
        popsize = 8
    else:
        popsize = 9

    if strategy_name in [
        "CMA_ES",
        "Differential_ES",
        "PSO_ES",
        "Simple_ES",
        "Simple_GA",
    ]:
        batch_eval = batch_rosenbrock
        fitness_shaper = FitnessShaper()
    elif strategy_name in ["Open_NES", "PEPG_ES"]:
        batch_eval = batch_quadratic
        fitness_shaper = FitnessShaper(z_score_fitness=True)

    strategy = Strat(popsize=popsize, num_dims=2)
    params = strategy.default_params
    state = strategy.initialize(rng, params)

    fitness_log = []
    for t in range(num_iters):
        rng, rng_iter = jax.random.split(rng)
        x, state = strategy.ask(rng_iter, state, params)
        fitness = batch_eval(x)
        fitness_shaped = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fitness_shaped, state, params)
        best_id = jnp.argmin(fitness)
        fitness_log.append(fitness[best_id])
    # assert fitness[0] >= fitness[-1]


def test_strategy_scan(strategy_name):
    # Loop over all strategies and test ask API
    rng = jax.random.PRNGKey(0)
    Strat = Strategies[strategy_name]
    # PBT also returns copy ID integer - treat separately
    if strategy_name in ["Persistent_ES", "Open_NES"]:
        popsize = 8
    else:
        popsize = 9

    if strategy_name in [
        "CMA_ES",
        "Differential_ES",
        "PSO_ES",
        "Simple_ES",
        "Simple_GA",
    ]:
        batch_eval = batch_rosenbrock
        fitness_shaper = FitnessShaper()
    elif strategy_name in ["Open_NES", "PEPG_ES"]:
        batch_eval = batch_quadratic
        fitness_shaper = FitnessShaper(z_score_fitness=True)

    strategy = Strat(popsize=popsize, num_dims=2)
    es_params = strategy.default_params

    @partial(jax.jit, static_argnums=(1,))
    def run_plain_es(rng, num_steps):
        """Run evolution ask-eval-tell loop."""
        state = strategy.initialize(rng, es_params)

        def step(state_input, tmp):
            """Helper function to lax.scan through."""
            rng, state = state_input
            rng, rng_iter = jax.random.split(rng)
            x, state = strategy.ask(rng_iter, state, es_params)
            fitness = batch_eval(x)
            fitness_shaped = fitness_shaper.apply(x, fitness)
            state = strategy.tell(x, fitness_shaped, state, es_params)
            best_id = jnp.argmin(fitness)
            return [rng, state], fitness[best_id]

        _, scan_out = jax.lax.scan(step, [rng, state], [jnp.zeros(num_steps)])
        return jnp.min(scan_out)

    run_plain_es(rng, num_iters)
