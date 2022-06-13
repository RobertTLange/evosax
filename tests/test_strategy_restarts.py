import jax
import jax.numpy as jnp
from evosax import OpenES, CMA_ES, IPOP_CMA_ES, BIPOP_CMA_ES
from evosax.restarts import Simple_Restarter, BIPOP_Restarter, IPOP_Restarter


def test_simple_restart():
    rng = jax.random.PRNGKey(0)
    strategy = CMA_ES(popsize=4, num_dims=2)
    re_strategy = Simple_Restarter(strategy)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.initialize(rng, re_es_params)

    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False

    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    x, state = re_strategy.ask(rng, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()


def test_ipop_restart():
    rng = jax.random.PRNGKey(0)
    strategy = OpenES(popsize=4, num_dims=2)
    re_strategy = IPOP_Restarter(strategy)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.initialize(rng, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_popsize == 4
    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    # Run 3rd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_popsize == 8

    fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_popsize == 8


def test_bipop_restart():
    rng = jax.random.PRNGKey(0)
    strategy = OpenES(popsize=4, num_dims=2)
    re_strategy = BIPOP_Restarter(strategy)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.initialize(rng, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_popsize == 4

    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    # Run 3rd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_popsize == 8

    fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_popsize == 8

    # Run 4th ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    x, state = re_strategy.ask(rng, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_popsize <= 8


def test_ipop_cma_es():
    rng = jax.random.PRNGKey(0)
    re_strategy = IPOP_CMA_ES(popsize=4, num_dims=2)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.initialize(rng, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_popsize == 4
    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True
    # Run 3rd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_popsize == 8

    # fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    # state = re_strategy.tell(x, fitness, state, re_es_params)
    # assert state.restart_state.restart_next == False
    # assert state.restart_state.active_popsize == 8


def test_bipop_cma_es():
    rng = jax.random.PRNGKey(0)
    re_strategy = BIPOP_CMA_ES(popsize=4, num_dims=2)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.initialize(rng, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_popsize == 4

    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    # Run 3rd ask-tell-generation
    x, state = re_strategy.ask(rng, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_popsize == 8

    # fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    # re_es_params = re_strategy.default_params
    # state = re_strategy.tell(x, fitness, state, re_es_params)
    # assert state.restart_state.restart_next == False
    # assert state.restart_state.active_popsize == 8

    # # Run 4th ask-tell-generation
    # x, state = re_strategy.ask(rng, state, re_es_params)
    # fitness = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # state = re_strategy.tell(x, fitness, state, re_es_params)
    # assert state.restart_state.restart_next == True

    # x, state = re_strategy.ask(rng, state, re_es_params)
    # assert (state.strategy_state.mean == jnp.zeros(2)).all()
    # assert state.restart_state.active_popsize <= 8
