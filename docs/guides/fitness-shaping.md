# Fitness shaping

Fitness shaping transforms the raw fitness values of a population before the algorithm
uses them. Replacing fitness values with ranks, for example, makes the update invariant
to the scale of the objective and less sensitive to outliers.

Every algorithm has a default shaping function suited to its update rule. Pass a
different one as `fitness_shaping_fn`:

```python
import jax.numpy as jnp
from evosax.algorithms import Open_ES
from evosax.core.fitness_shaping import standardize_fitness_shaping_fn

es = Open_ES(
    population_size=16,
    solution=jnp.zeros(4),
    fitness_shaping_fn=standardize_fitness_shaping_fn,
)
```

## Built-in functions

All of these live in `evosax.core.fitness_shaping`.

| Function | Shaped fitness |
| --- | --- |
| `identity_fitness_shaping_fn` | The raw fitness. |
| `standardize_fitness_shaping_fn` | Zero mean and unit variance (z-score). |
| `normalize_fitness_shaping_fn` | Linearly rescaled to `[-1, 1]`. |
| `centered_rank_fitness_shaping_fn` | Ranks rescaled to `[-0.5, 0.5]`. |
| `weights_fitness_shaping_fn` | Rank-based recombination weights taken from `params.weights`. |

`weights_fitness_shaping_fn` is the default of CMA-ES and related strategies and
requires hyperparameters with a `weights` field. These algorithms expect weights rather
than fitness values, so their shaping function should normally be left unchanged.

The shaped fitness is only used for the update. `state.best_fitness` and the default
metrics always report raw fitness.

## Weight decay

`add_weight_decay` wraps any shaping function and adds an L2 penalty on the solution to
the raw fitness first:

```python
from evosax.core.fitness_shaping import add_weight_decay

fitness_shaping_fn = add_weight_decay(standardize_fitness_shaping_fn, weight_decay=0.01)
```

## Writing your own

A fitness shaping function takes the flat population, the raw fitness, and the state and
params of the algorithm, and returns the shaped fitness. For algorithms that do not use
recombination weights, lower values must remain better:

```python
def clipped_fitness_shaping_fn(population, fitness, state, params):
    return jnp.clip(fitness, -10.0, 10.0)


es = Open_ES(
    population_size=16,
    solution=jnp.zeros(4),
    fitness_shaping_fn=clipped_fitness_shaping_fn,
)
```

See the [API reference](../reference/core/fitness_shaping.md) for details, including
how each function treats NaN values.
