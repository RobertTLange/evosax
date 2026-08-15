"""Integration tests for restart controllers."""

import jax
import jax.numpy as jnp
import pytest
from evosax.algorithms import CMA_ES, Open_ES
from evosax.core.restart import (
    BIPOPRestart,
    BIPOPRestartParams,
    BIPOPRestartState,
    IPOPRestart,
    IPOPRestartParams,
    RestartParams,
    SimpleRestart,
    cma_cond,
    fitness_std_cond,
    generation_cond,
    spread_cond,
)
from flax import serialization


def always_restart(*_args):
    """Request a restart for controller tests."""
    return jnp.array(True)


def open_es_factory(population_size):
    """Create a small OpenAI-ES instance for restart tests."""
    return Open_ES(population_size=population_size, solution=jnp.zeros(2))


def cma_es_factory(population_size):
    """Create a small CMA-ES instance for restart tests."""
    return CMA_ES(population_size=population_size, solution=jnp.zeros(2))


def cma_restart_params(strategy, previous_params):
    """Retain CMA-ES standard-deviation configuration across a restart."""
    return strategy.default_params.replace(std_init=previous_params.std_init)


def even_population_size(population_size):
    """Round population sizes up for antithetic OpenAI-ES sampling."""
    return population_size + population_size % 2


def test_simple_restart_waits_for_minimum_generation_count():
    strategy = open_es_factory(4)
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)
    restarter = SimpleRestart(
        stop_criteria=(always_restart,), restart_params=RestartParams(min_num_gens=2)
    )
    restart_state = restarter.init()

    assert not bool(
        restarter.should_restart(
            None, jnp.zeros(4), state, strategy.default_params, restart_state
        )
    )
    assert bool(
        restarter.should_restart(
            None,
            jnp.zeros(4),
            state.replace(generation_counter=2),
            strategy.default_params,
            restart_state,
        )
    )


def test_simple_restart_copies_mean_and_archive_when_requested():
    strategy = open_es_factory(4)
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)
    state = state.replace(
        mean=jnp.array([1.0, 2.0]),
        best_solution=jnp.array([3.0, 4.0]),
        best_fitness=-5.0,
    )
    restarter = SimpleRestart(restart_params=RestartParams(copy_mean=True))

    restarted_state, restart_state = restarter.restart(
        jax.random.key(1), strategy, state, strategy.default_params, restarter.init()
    )

    assert jnp.array_equal(restarted_state.mean, state.mean)
    assert jnp.array_equal(restarted_state.best_solution, state.best_solution)
    assert restarted_state.best_fitness == state.best_fitness
    assert restarted_state.generation_counter == 0
    assert restart_state.restart_counter == 1


def test_simple_restart_resets_mean_when_not_copying_it():
    strategy = open_es_factory(4)
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)
    state = state.replace(mean=jnp.array([1.0, 2.0]))
    restarter = SimpleRestart(restart_params=RestartParams(copy_mean=False))

    restarted_state, _ = restarter.restart(
        jax.random.key(1), strategy, state, strategy.default_params, restarter.init()
    )

    assert jnp.array_equal(restarted_state.mean, jnp.zeros(2))


def test_simple_restart_works_after_an_ask_tell_cycle():
    strategy = open_es_factory(4)
    params = strategy.default_params
    state = strategy.init(jax.random.key(0), jnp.zeros(2), params)
    population, state = strategy.ask(jax.random.key(1), state, params)
    state, _ = strategy.tell(jax.random.key(2), population, jnp.zeros(4), state, params)
    restarter = SimpleRestart(stop_criteria=(fitness_std_cond,))
    restart_state = restarter.init()

    assert bool(
        restarter.should_restart(population, jnp.zeros(4), state, params, restart_state)
    )

    restarted_state, restart_state = restarter.restart(
        jax.random.key(3), strategy, state, params, restart_state
    )

    assert restarted_state.generation_counter == 0
    assert restart_state.restart_counter == 1


def test_ipop_restart_rebuilds_strategy_with_a_larger_population():
    strategy = open_es_factory(4)
    params = strategy.default_params
    state = strategy.init(jax.random.key(0), jnp.ones(2), params)
    state = state.replace(best_solution=jnp.array([3.0, 4.0]), best_fitness=-5.0)
    restarter = IPOPRestart(
        open_es_factory,
        initial_population_size=4,
        restart_params=IPOPRestartParams(copy_mean=True),
    )

    strategy, params, state, restart_state = restarter.restart(
        jax.random.key(1), strategy, state, params, restarter.init(strategy)
    )
    population, _ = strategy.ask(jax.random.key(2), state, params)

    assert strategy.population_size == 8
    assert population.shape == (8, 2)
    assert restart_state.active_population_size == 8
    assert restart_state.restart_counter == 1
    assert jnp.array_equal(state.mean, jnp.ones(2))
    assert jnp.array_equal(state.best_solution, jnp.array([3.0, 4.0]))


def test_ipop_restart_uses_parameter_factory_for_resized_strategy():
    strategy = cma_es_factory(4)
    params = strategy.default_params.replace(std_init=0.25)
    state = strategy.init(jax.random.key(0), jnp.zeros(2), params)
    restarter = IPOPRestart(
        cma_es_factory,
        initial_population_size=4,
        strategy_params_factory=cma_restart_params,
    )

    _, params, _, _ = restarter.restart(
        jax.random.key(1), strategy, state, params, restarter.init(strategy)
    )

    assert params.std_init == 0.25


def test_population_restarters_reject_an_initial_population_size_mismatch():
    restarter = IPOPRestart(open_es_factory, initial_population_size=4)

    with pytest.raises(ValueError, match="initial_population_size"):
        restarter.init(open_es_factory(8))


def test_bipop_restart_excludes_initial_run_from_regime_budgets():
    strategy = cma_es_factory(4)
    params = strategy.default_params
    state = strategy.init(jax.random.key(0), jnp.zeros(2), params)
    state = state.replace(generation_counter=3)
    restarter = BIPOPRestart(cma_es_factory, initial_population_size=4)

    _, _, _, restart_state = restarter.restart(
        jax.random.key(1), strategy, state, params, restarter.init(strategy)
    )

    assert restart_state.large_eval_budget == 0
    assert restart_state.small_eval_budget == 0


def test_bipop_restart_state_keeps_its_existing_constructor_contract():
    restart_state = BIPOPRestartState(
        restart_counter=0,
        restart_next=False,
        active_population_size=4,
        restart_large_counter=0,
        large_eval_budget=0,
        small_eval_budget=0,
        small_pop_active=True,
    )

    assert jnp.isnan(restart_state.initial_std)


def test_bipop_restart_state_restores_legacy_state_dicts():
    restart_state = BIPOPRestartState(
        restart_counter=0,
        restart_next=False,
        active_population_size=4,
        restart_large_counter=0,
        large_eval_budget=0,
        small_eval_budget=0,
        small_pop_active=True,
    )
    legacy_state_dict = serialization.to_state_dict(restart_state)
    legacy_state_dict.pop("initial_std")

    restored_state = serialization.from_state_dict(restart_state, legacy_state_dict)

    assert jnp.isnan(restored_state.initial_std)


def test_bipop_restart_switches_from_large_to_small_population_by_budget():
    strategy = cma_es_factory(4)
    params = strategy.default_params
    state = strategy.init(jax.random.key(0), jnp.zeros(2), params)
    restarter = BIPOPRestart(
        cma_es_factory, initial_population_size=4, restart_params=BIPOPRestartParams()
    )

    strategy, params, state, restart_state = restarter.restart(
        jax.random.key(1), strategy, state, params, restarter.init(strategy)
    )
    state = state.replace(generation_counter=3)
    strategy, params, state, restart_state = restarter.restart(
        jax.random.key(0), strategy, state, params, restart_state
    )
    population, _ = strategy.ask(jax.random.key(3), state, params)

    assert restart_state.restart_counter == 2
    assert restart_state.small_pop_active
    assert restart_state.large_eval_budget == 24
    assert population.shape == (strategy.population_size, 2)
    assert 4 <= strategy.population_size < 8


def test_bipop_small_population_resamples_cma_initial_standard_deviation():
    strategy = cma_es_factory(4)
    params = strategy.default_params
    state = strategy.init(jax.random.key(0), jnp.zeros(2), params)
    restarter = BIPOPRestart(cma_es_factory, initial_population_size=4)

    strategy, params, state, restart_state = restarter.restart(
        jax.random.key(1), strategy, state, params, restarter.init(strategy)
    )
    state = state.replace(generation_counter=3)
    _, params, _, restart_state = restarter.restart(
        jax.random.key(2), strategy, state, params, restart_state
    )

    assert restart_state.small_pop_active
    assert 0.01 <= params.std_init < 1.0


def test_bipop_small_populations_use_the_initial_standard_deviation():
    strategy = cma_es_factory(4)
    params = strategy.default_params.replace(std_init=0.5)
    state = strategy.init(jax.random.key(0), jnp.zeros(2), params)
    restarter = BIPOPRestart(
        cma_es_factory,
        initial_population_size=4,
        strategy_params_factory=cma_restart_params,
    )

    strategy, params, state, restart_state = restarter.restart(
        jax.random.key(1), strategy, state, params, restarter.init(strategy)
    )
    state = state.replace(generation_counter=3)
    strategy, params, state, restart_state = restarter.restart(
        jax.random.key(2), strategy, state, params, restart_state
    )
    state = state.replace(generation_counter=1)
    _, params, _, restart_state = restarter.restart(
        jax.random.key(3), strategy, state, params, restart_state
    )

    assert restart_state.small_pop_active
    assert 0.005 <= params.std_init < 0.5


def test_bipop_restart_applies_population_size_transform_before_rebuilding():
    strategy = open_es_factory(4)
    params = strategy.default_params
    state = strategy.init(jax.random.key(0), jnp.zeros(2), params)
    restarter = BIPOPRestart(
        open_es_factory,
        initial_population_size=4,
        population_size_transform=even_population_size,
    )

    strategy, params, state, restart_state = restarter.restart(
        jax.random.key(1), strategy, state, params, restarter.init(strategy)
    )
    state = state.replace(generation_counter=3)
    strategy, _, _, restart_state = restarter.restart(
        jax.random.key(2), strategy, state, params, restart_state
    )

    assert strategy.population_size % 2 == 0
    assert restart_state.active_population_size == strategy.population_size


def test_bipop_small_population_uses_the_current_large_regime():
    strategy = cma_es_factory(4)
    params = strategy.default_params
    state = strategy.init(jax.random.key(0), jnp.zeros(2), params)
    restarter = BIPOPRestart(
        cma_es_factory,
        initial_population_size=4,
        restart_params=BIPOPRestartParams(population_size_multiplier=3),
    )

    strategy, params, state, restart_state = restarter.restart(
        jax.random.key(1), strategy, state, params, restarter.init(strategy)
    )
    state = state.replace(generation_counter=3)
    strategy, _, _, restart_state = restarter.restart(
        jax.random.key(0), strategy, state, params, restart_state
    )

    assert restart_state.small_pop_active
    assert 4 <= strategy.population_size < 6


def test_cma_condition_accepts_current_cma_es_state():
    strategy = CMA_ES(population_size=4, solution=jnp.zeros(2))
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)

    assert not bool(
        cma_cond(
            None, jnp.zeros(4), state, strategy.default_params, None, RestartParams()
        )
    )


def test_cma_condition_uses_standard_deviation_of_covariance_diagonal():
    strategy = CMA_ES(population_size=4, solution=jnp.zeros(2))
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)
    state = state.replace(C=jnp.eye(2) * 1e-16)

    assert not bool(
        cma_cond(
            None, jnp.zeros(4), state, strategy.default_params, None, RestartParams()
        )
    )


def test_cma_condition_detects_an_ill_conditioned_covariance_matrix():
    strategy = CMA_ES(population_size=4, solution=jnp.zeros(2))
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)
    state = state.replace(C=jnp.diag(jnp.array([1e-8, 1e8])))

    assert bool(
        cma_cond(
            None, jnp.zeros(4), state, strategy.default_params, None, RestartParams()
        )
    )


def test_cma_condition_requires_small_absolute_evolution_path():
    strategy = CMA_ES(population_size=4, solution=jnp.zeros(2))
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)
    state = state.replace(C=jnp.eye(2) * 1e-30, p_c=jnp.full(2, -1e6))

    assert not bool(
        cma_cond(
            None, jnp.zeros(4), state, strategy.default_params, None, RestartParams()
        )
    )


def test_restart_controller_rejects_batched_distribution_state():
    strategy = open_es_factory(4)
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)
    state = state.replace(mean=jnp.zeros((2, 2)))
    restarter = SimpleRestart(stop_criteria=(always_restart,))

    with pytest.raises(ValueError, match="single-distribution"):
        restarter.should_restart(
            None, jnp.zeros(4), state, strategy.default_params, restarter.init()
        )


def test_generation_and_spread_conditions_use_restart_parameters():
    strategy = open_es_factory(4)
    state = strategy.init(jax.random.key(0), jnp.zeros(2), strategy.default_params)
    state = state.replace(generation_counter=3)
    restart_params = RestartParams(
        generation_threshold=3,
        min_fitness_spread=0.5,
    )

    assert bool(
        generation_cond(
            None,
            jnp.zeros(4),
            state,
            strategy.default_params,
            None,
            restart_params,
        )
    )
    assert bool(
        spread_cond(
            None,
            jnp.array([1.0, 1.4]),
            state,
            strategy.default_params,
            None,
            restart_params,
        )
    )
