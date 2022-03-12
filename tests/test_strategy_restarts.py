import jax
import jax.numpy as jnp
from evosax import CMA_ES
from evosax.restarts import Simple_Restarter


def test_simple_restart():
    rng = jax.random.PRNGKey(0)
    strategy = CMA_ES(popsize=4, num_dims=2)
    re_strategy = Simple_Restarter(strategy)
    re_es_params = re_strategy.default_params
    re_es_params["min_num_gens"] = 2
    state = re_strategy.initialize(rng, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(rng, x, fitness, state, re_es_params)
    assert state["restarted"] == False

    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(rng, x, fitness, state, re_es_params)
    assert state["restarted"] == True
    assert (state["mean"] == jnp.zeros(2)).all()
