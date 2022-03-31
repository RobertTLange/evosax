import jax
import jax.numpy as jnp
from evosax.subpops import BatchStrategy


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
