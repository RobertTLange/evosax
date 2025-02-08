import jax
import jax.numpy as jnp
from evosax import BIPOP_CMA_ES, CMA_ES, IPOP_CMA_ES, OpenES
from evosax.restarts import BIPOP_Restarter, IPOP_Restarter, Simple_Restarter


def test_simple_restart():
    num_dims = 2
    x = jnp.zeros((num_dims,))

    key = jax.random.key(0)
    strategy = CMA_ES(population_size=4, pholder_params=x)
    re_strategy = Simple_Restarter(strategy)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.init(key, re_es_params)

    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False

    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    x, state = re_strategy.ask(key, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()


def test_ipop_restart():
    key = jax.random.key(0)
    num_dims = 2
    x = jnp.zeros((num_dims,))
    strategy = OpenES(population_size=4, pholder_params=x)
    re_strategy = IPOP_Restarter(strategy)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.init(key, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_population_size == 4
    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    # Run 3rd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_population_size == 8

    fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_population_size == 8


def test_bipop_restart():
    key = jax.random.key(0)
    num_dims = 2
    x = jnp.zeros((num_dims,))
    strategy = OpenES(population_size=4, pholder_params=x)
    re_strategy = BIPOP_Restarter(strategy)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.init(key, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_population_size == 4

    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    # Run 3rd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_population_size == 8

    fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_population_size == 8

    # Run 4th ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    x, state = re_strategy.ask(key, state, re_es_params)
    assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_population_size <= 8


def test_ipop_cma_es():
    key = jax.random.key(0)
    num_dims = 2
    x = jnp.zeros((num_dims,))
    re_strategy = IPOP_CMA_ES(population_size=4, pholder_params=x)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.init(key, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_population_size == 4
    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True
    # Run 3rd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    # assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_population_size == 8

    fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_population_size == 8


def test_bipop_cma_es():
    key = jax.random.key(0)
    num_dims = 2
    x = jnp.zeros((num_dims,))
    re_strategy = BIPOP_CMA_ES(population_size=4, pholder_params=x)
    re_es_params = re_strategy.default_params
    re_es_params = re_es_params.replace(
        restart_params=re_es_params.restart_params.replace(min_num_gens=2)
    )
    state = re_strategy.init(key, re_es_params)
    # Run 1st ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_population_size == 4

    # Run 2nd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    # Run 3rd ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    # assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_population_size == 8

    fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == False
    assert state.restart_state.active_population_size == 8

    # Run 4th ask-tell-generation
    x, state = re_strategy.ask(key, state, re_es_params)
    fitness = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = re_strategy.tell(x, fitness, state, re_es_params)
    assert state.restart_state.restart_next == True

    x, state = re_strategy.ask(key, state, re_es_params)
    # assert (state.strategy_state.mean == jnp.zeros(2)).all()
    assert state.restart_state.active_population_size <= 8
