# Concepts

## Ask, eval, tell

An evosax algorithm never calls your objective function. Instead it hands you candidate
solutions and you report how good they are:

```python
population, state = es.ask(key, state, params)
fitness = ...  # evaluate the population however you like
state, metrics = es.tell(key, population, fitness, state, params)
```

This keeps the evaluation entirely under your control. It can be a `jax.vmap` over a
loss function, a batch of simulator rollouts, or one of the built-in
[problems](problems.md).

**Fitness is minimized.** `tell` treats lower fitness as better. To maximize, pass the
negative of your objective. A NaN fitness marks a failed evaluation: it is never
recorded as the best solution.

## State and params

Algorithms are stateless Python objects. Everything that changes during a run lives in
a `state` dataclass, and every tunable hyperparameter lives in a `params` dataclass.
Both are immutable PyTrees that you pass in and get back explicitly.

- `es.default_params` returns the default hyperparameters.
- `es.init(key, ..., params)` creates the initial state.
- `state.best_solution`, `state.best_fitness` and `state.generation_counter` are
  available for every algorithm.

To change a hyperparameter, create a modified copy with `replace`:

```python
import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES

es = CMA_ES(population_size=16, solution=jnp.zeros(4))
params = es.default_params.replace(std_init=0.5)
state = es.init(jax.random.key(0), jnp.zeros(4), params)
```

Because `state` and `params` are ordinary JAX PyTrees, `init`, `ask` and `tell` can be
used inside `jax.jit`, `jax.lax.scan` and `jax.vmap`. Vectorizing over `params` runs a
batch of hyperparameter settings in parallel; vectorizing over keys runs independent
seeds.

## Constructor arguments

Settings that determine array shapes or the structure of the computation are passed to
the constructor rather than stored in `params`:

- `population_size`: the number of candidates per generation.
- `solution`: an example solution. Only its structure and shapes are used.
- `fitness_shaping_fn`: how raw fitness is transformed before the update, see
  [Fitness shaping](fitness-shaping.md).
- `metrics_fn`: what `tell` returns as `metrics`.
- Algorithm-specific arguments. Many gradient-style strategies accept an Optax
  `optimizer` and a `std_schedule` for the exploration noise:

```python
import optax
from evosax.algorithms import Open_ES

es = Open_ES(
    population_size=64,
    solution=jnp.zeros(4),
    optimizer=optax.adam(learning_rate=0.01),
    std_schedule=optax.exponential_decay(0.1, transition_steps=100, decay_rate=0.9),
)
```

evosax also ships the ClipUp optimizer as `evosax.core.optimizer.clipup`.

## Solutions are PyTrees

A solution can be an array or any nested structure of arrays, such as neural network
parameters. Algorithms flatten solutions internally and return populations in the
original structure, with a leading population axis on every leaf. State fields such as
`state.mean` and `state.best_solution` are flat vectors.

## Two families of algorithms

**Distribution-based** algorithms, such as CMA-ES, OpenAI-ES or SNES, maintain a search
distribution. `ask` samples from it and `tell` updates it. They are initialized from a
starting mean:

```python
state = es.init(key, mean, params)
```

`es.get_mean(state)` returns the current mean as a solution PyTree.

**Population-based** algorithms, such as Differential Evolution, PSO or the genetic
algorithms, keep an explicit population. They are initialized from a population that has
already been evaluated:

```python
state = ga.init(key, population, fitness, params)
```

`ga.get_population(state)` and `ga.get_best_solution(state)` return the stored
population and its best member as PyTrees.

The [algorithm overview](../algorithms.md) shows which family each algorithm belongs to.

## Selecting an algorithm by name

All algorithms are also available in a dictionary, which is convenient for sweeps and
configuration files:

```python
from evosax.algorithms import algorithms

ES = algorithms["Sep_CMA_ES"]
es = ES(population_size=16, solution=jnp.zeros(4))
```

## Custom metrics

`metrics_fn` receives `(key, population, fitness, state, params)` and returns a PyTree.
Extend the default to log additional parts of the state:

```python
from evosax.algorithms import Sep_CMA_ES
from evosax.algorithms.distribution_based.base import metrics_fn


def custom_metrics_fn(key, population, fitness, state, params):
    metrics = metrics_fn(key, population, fitness, state, params)
    return metrics | {"std": state.std}


es = Sep_CMA_ES(population_size=16, solution=jnp.zeros(4), metrics_fn=custom_metrics_fn)
```
