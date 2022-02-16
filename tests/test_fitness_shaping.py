import jax.numpy as jnp
from evosax import FitnessShaper


def test_fitness_shaping_rank():
    shaper = FitnessShaper(centered_rank=True)
    x = jnp.array([[1.0], [2.0], [3.0]])
    fit = jnp.array([0.0, 1.0, 2.0])
    fit_re = shaper.apply(x, fit)
    assert jnp.allclose(fit_re, jnp.array([-0.5, 0.0, 0.5]), atol=1e-03)


def test_fitness_shaping_decay():
    shaper = FitnessShaper(w_decay=0.01)
    x = jnp.array([[1.0], [2.0], [3.0]])
    fit = jnp.array([0.0, 1.0, 2.0])
    fit_re = shaper.apply(x, fit)
    assert jnp.allclose(
        fit_re, jnp.array([0.01, 1 + 0.04, 2 + 0.09]), atol=1e-03
    )


def test_fitness_shaping_zscore():
    shaper = FitnessShaper(z_score=True)
    x = jnp.array([[1.0], [2.0], [3.0]])
    fit = jnp.array([0.0, 1.0, 2.0])
    fit_re = shaper.apply(x, fit)
    assert jnp.allclose(
        fit_re, jnp.array([-1.2247448, 0.0, 1.2247448]), atol=1e-03
    )


def test_fitness_shaping_max():
    shaper = FitnessShaper(w_decay=0.01, maximize=True)
    x = jnp.array([[1.0], [2.0], [3.0]])
    fit = jnp.array([0.0, 1.0, 2.0])
    fit_re = shaper.apply(x, fit)
    assert jnp.allclose(
        fit_re, jnp.array([-0.01, -(1 + 0.04), -(2 + 0.09)]), atol=1e-03
    )
