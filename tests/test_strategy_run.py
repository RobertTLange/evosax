import jax
import jax.numpy as jnp
from evosax import Strategies
from evosax.problems import batch_rosenbrock, batch_quadratic
from evosax.utils import FitnessShaper


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
