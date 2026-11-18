# Quickstart

This page runs CMA-ES on a benchmark function, compiles the optimization loop, and then
optimizes a PyTree of parameters with a population-based algorithm.

## The ask-eval-tell loop

Every algorithm in evosax is driven by the same three steps:

1. `ask` the algorithm for a population of candidate solutions,
2. evaluate the fitness of each candidate,
3. `tell` the algorithm the fitness values so it can update its state.

Start with a problem. `BBOBProblem` provides the functions of the black-box optimization
benchmark:

```python
import jax
import jax.numpy as jnp
from evosax.problems import BBOBProblem

key = jax.random.key(0)

problem = BBOBProblem(fn_name="rosenbrock", num_dims=2, f_opt=0.0)

key, subkey = jax.random.split(key)
problem_state = problem.init(subkey)

key, subkey = jax.random.split(key)
solution = problem.sample(subkey)  # a point in the search space
```

Create the algorithm. The `solution` argument is an example solution: only its structure
and shapes matter, they tell the algorithm what the search space looks like.

```python
from evosax.algorithms import CMA_ES

es = CMA_ES(population_size=16, solution=solution)
params = es.default_params

key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)  # start the search at `solution`
```

Then run the loop:

```python
for _ in range(64):
    key, key_ask, key_eval, key_tell = jax.random.split(key, 4)

    population, state = es.ask(key_ask, state, params)
    fitness, problem_state, _ = problem.eval(key_eval, population, problem_state)
    state, metrics = es.tell(key_tell, population, fitness, state, params)

print(state.best_fitness)
```

`state.best_solution` and `state.best_fitness` hold the best candidate evaluated so far.
`metrics` is a dictionary of diagnostics for the current generation, such as
`best_fitness` and, for distribution-based algorithms, the `mean` of the search
distribution.

!!! note "evosax minimizes"

    Lower fitness is better. If your objective should be maximized, such as a reward,
    pass `-fitness` to `tell`.

## Compiling the loop

`ask` and `tell` are pure functions, so a whole run can be compiled with
`jax.lax.scan`:

```python
def step(carry, key):
    state, problem_state = carry
    key_ask, key_eval, key_tell = jax.random.split(key, 3)

    population, state = es.ask(key_ask, state, params)
    fitness, problem_state, _ = problem.eval(key_eval, population, problem_state)
    state, metrics = es.tell(key_tell, population, fitness, state, params)

    return (state, problem_state), metrics


key, key_init, key_run = jax.random.split(key, 3)
state = es.init(key_init, solution, params)

keys = jax.random.split(key_run, 64)
(state, problem_state), metrics = jax.lax.scan(step, (state, problem_state), keys)

print(metrics["best_fitness"].shape)  # (64,): one entry per generation
```

Because a run is now a single JAX function, it can also be vectorized, for example
over random seeds with `jax.vmap`.

## Optimizing a PyTree

Solutions do not have to be flat vectors. Any PyTree works, for example the parameters
of a neural network. The population returned by `ask` has the same structure with an
additional leading axis, so the fitness function can be applied with `jax.vmap`:

```python
import optax
from evosax.algorithms import Open_ES

solution = {"w": jnp.zeros((3, 2)), "b": jnp.zeros(2)}


def loss(p):
    return jnp.sum((p["w"] - 1.0) ** 2) + jnp.sum((p["b"] + 1.0) ** 2)


es = Open_ES(
    population_size=64,
    solution=solution,
    optimizer=optax.adam(learning_rate=0.1),
    std_schedule=optax.constant_schedule(0.1),
)
params = es.default_params

key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)

for _ in range(200):
    key, key_ask, key_tell = jax.random.split(key, 3)
    population, state = es.ask(key_ask, state, params)  # {"w": (64, 3, 2), "b": (64, 2)}
    fitness = jax.vmap(loss)(population)
    state, metrics = es.tell(key_tell, population, fitness, state, params)

mean = es.get_mean(state)  # same structure as `solution`
```

Internally the algorithm works on flat vectors, so `state.mean` and
`state.best_solution` are flat. `get_mean` converts the mean back into a PyTree.

## Population-based algorithms

Algorithms such as Differential Evolution or the genetic algorithms keep an explicit
population instead of a search distribution. They use the same loop, but are initialized
from a population that has already been evaluated:

```python
from evosax.algorithms import DifferentialEvolution


def sphere(x):
    return jnp.sum(x**2)


solution = jnp.zeros(8)
de = DifferentialEvolution(population_size=32, solution=solution)
params = de.default_params

key, key_population, key_init = jax.random.split(key, 3)
population = jax.random.normal(key_population, (32, 8))
fitness = jax.vmap(sphere)(population)
state = de.init(key_init, population, fitness, params)

for _ in range(100):
    key, key_ask, key_tell = jax.random.split(key, 3)
    population, state = de.ask(key_ask, state, params)
    fitness = jax.vmap(sphere)(population)
    state, metrics = de.tell(key_tell, population, fitness, state, params)

print(state.best_fitness)
```

## Next steps

- [Concepts](../guides/concepts.md) covers state, params and how to configure an
  algorithm.
- [Algorithms](../algorithms.md) lists everything that is implemented.
- The [example notebooks](../examples.md) cover reinforcement learning, vision, restarts
  and more.
