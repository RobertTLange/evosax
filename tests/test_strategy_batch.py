import jax
import jax.numpy as jnp
from evosax.subpops import BatchStrategy, MetaStrategy


def test_batch_strategy():
    rng = jax.random.PRNGKey(0)
    batch_strategy = BatchStrategy(
        strategy_name="CMA_ES",
        num_dims=2,
        popsize=100,
        num_subpops=5,
        strategy_kwargs={"elite_ratio": 0.5},
    )
    es_params = batch_strategy.default_params
    state = batch_strategy.initialize(rng, es_params)
    assert state["mean"].shape == (5, 2)

    x, state = batch_strategy.ask(rng, state, es_params)
    assert x.shape == (100, 2)

    fitness = jnp.zeros(100)
    state = batch_strategy.tell(x, fitness, state, es_params)
    assert state["mean"].shape == (5, 2)


def test_meta_strategy():
    rng = jax.random.PRNGKey(0)
    meta_strategy = MetaStrategy(
        meta_strategy_name="CMA_ES",
        inner_strategy_name="DE",
        meta_params=["diff_w", "cross_over_rate"],
        num_dims=2,
        popsize=100,
        num_subpops=10,
        meta_strategy_kwargs={"elite_ratio": 0.5},
    )
    meta_es_params = meta_strategy.default_params_meta
    meta_es_params["clip_min"] = jnp.array([0, 0])
    meta_es_params["clip_max"] = jnp.array([2, 1])

    # META: Initialize the meta strategy state
    inner_es_params = meta_strategy.default_params
    meta_state = meta_strategy.initialize_meta(rng, meta_es_params)

    # META: Get altered inner es hyperparams
    inner_es_params, meta_state = meta_strategy.ask_meta(
        rng, meta_state, meta_es_params, inner_es_params
    )
    assert meta_state["mean"].shape == (2,)

    # INNER: Initialize the inner batch ES
    state = meta_strategy.initialize(rng, inner_es_params)
    assert state["mean"].shape == (10, 2)

    # INNER: Ask for inner candidate params to evaluate on problem
    x, state = meta_strategy.ask(rng, state, inner_es_params)
    assert x.shape == (100, 2)

    # INNER: Update using pseudo fitness
    fitness = jax.random.normal(rng, (100,))
    state = meta_strategy.tell(x, fitness, state, inner_es_params)
    assert state["mean"].shape == (10, 2)

    # META: Update the meta strategy
    meta_state = meta_strategy.tell_meta(
        inner_es_params, fitness, meta_state, meta_es_params
    )
    assert meta_state["mean"].shape == (2,)
