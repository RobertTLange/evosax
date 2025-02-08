import jax.numpy as jnp
from evosax.utils import ESLog


def test_es_log():
    num_dims = 2
    x = jnp.zeros((num_dims,))
    es_logging = ESLog(pholder_params=x, num_generations=10, top_k=3, maximize=True)
    log = es_logging.init()
    x = jnp.array([[1, 2], [2, 4], [4, 6], [6, 7]])
    fitness = jnp.array([1, 2, 3, 4])
    log = es_logging.update(log, x, fitness)
    assert log["log_top_1"][0] == 4
    assert log["top_params"][0][0] == 6
    assert log["top_params"][0][1] == 7


def test_top_k():
    num_dims = 2
    x = jnp.zeros((num_dims,))
    es_logging = ESLog(pholder_params=x, num_generations=10, top_k=3, maximize=True)
    log = es_logging.init()
    x = jnp.array([[1, 2], [2, 4], [4, 6], [6, 7]])
    fitness = jnp.array([1, 2, 3, 4])
    log = es_logging.update(log, x, fitness)
    assert jnp.array_equal(log["top_fitness"], jnp.array([4, 3, 2]))
    assert jnp.array_equal(log["top_params"][0], jnp.array([6, 7]))
    assert jnp.array_equal(log["top_params"][1], jnp.array([4, 6]))
    assert jnp.array_equal(log["top_params"][2], jnp.array([2, 4]))
