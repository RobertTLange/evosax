import jax.numpy as jnp
from evosax.utils import ESLog


def test_es_log():
    es_logging = ESLog(num_dims=2, num_generations=10, top_k=3, maximize=True)
    log = es_logging.initialize()
    x = jnp.array([[1, 2], [2, 4], [4, 6], [6, 7]])
    fitness = jnp.array([1, 2, 3, 4])
    log = es_logging.update(log, x, fitness)
    assert log["log_top_1"][0] == 4
    assert log["top_params"][0][0] == 6
    assert log["top_params"][0][1] == 7
