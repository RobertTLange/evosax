from functools import partial

import jax
import jax.numpy as jnp
from evosax import Strategies
from evosax.core import FitnessShaper
from evosax.problems import BBOBFitness

num_iters = 25


def test_strategy_run(strategy_name):
    # Loop over all strategies and test ask API
    rng = jax.random.PRNGKey(0)
    Strat = Strategies[strategy_name]

    num_dims = 2
    x = jnp.zeros((num_dims,))

    # PBT also returns copy ID integer - treat separately
    if strategy_name == "ESMC":
        popsize = 21
    else:
        popsize = 20
    if strategy_name in ["SV_CMA_ES", "SV_OpenAI_ES", "SV_OpenES"]:
        strategy = Strat(npop=1, subpopsize=popsize, pholder_params=x)
    else:
        strategy = Strat(popsize=popsize, pholder_params=x)
    evaluator = BBOBFitness("sphere", 2)
    fitness_shaper = FitnessShaper()

    batch_eval = evaluator.rollout
    params = strategy.default_params
    state = strategy.initialize(rng, params)

    fitness_log = []
    for t in range(num_iters):
        rng, rng_eval, rng_iter = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_iter, state, params)
        fitness = batch_eval(rng, x)
        fitness_shaped = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fitness_shaped, state, params)
        best_id = jnp.argmin(fitness)
        fitness_log.append(fitness[best_id])
    # assert fitness[0] >= fitness[-1]


def test_strategy_scan(strategy_name):
    # Loop over all strategies and test ask API
    rng = jax.random.PRNGKey(0)
    Strat = Strategies[strategy_name]

    num_dims = 2
    x = jnp.zeros((num_dims,))

    # PBT also returns copy ID integer - treat separately
    if strategy_name == "ESMC":
        popsize = 21
    else:
        popsize = 20
    if strategy_name in ["SV_CMA_ES", "SV_OpenAI_ES", "SV_OpenES"]:
        strategy = Strat(npop=1, subpopsize=popsize, pholder_params=x)
    elif strategy_name in ["BIPOP_CMA_ES", "IPOP_CMA_ES"]:
        return
    else:
        strategy = Strat(popsize=popsize, pholder_params=x)
    evaluator = BBOBFitness("sphere", 2)
    fitness_shaper = FitnessShaper()

    batch_eval = evaluator.rollout
    es_params = strategy.default_params

    state = strategy.initialize(rng, es_params)

    def step(carry, _):
        """Helper function to lax.scan."""
        rng, state = carry
        rng, rng_eval, rng_iter = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_iter, state, es_params)
        fitness = batch_eval(rng_eval, x)
        fitness_shaped = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fitness_shaped, state, es_params)
        return (rng, state), jnp.min(fitness)

    _, best_fitness = jax.lax.scan(
        step,
        init=(rng, state),
        length=num_iters,
    )
