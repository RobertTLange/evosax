# Restarts

Evolution strategies can converge prematurely, for example to a local optimum. A restart
controller watches a run and, once it has converged, reinitializes the strategy while
keeping the best solution found so far.

evosax provides three controllers in `evosax.core.restart`:

| Controller | Behaviour |
| --- | --- |
| `SimpleRestart` | Restarts with the same population size. |
| `IPOPRestart` | Multiplies the population size at every restart. |
| `BIPOPRestart` | Alternates between small and large populations by evaluation budget. |

Restart controllers work with distribution-based strategies that maintain a single
search distribution, so not with the Stein Variational strategies.

## Usage

A controller is used in the Python loop around `ask` and `tell`. After each generation,
`should_restart` checks the stop criteria and `restart` reinitializes the strategy.

`IPOPRestart` and `BIPOPRestart` change the population size, which changes array shapes.
They therefore take a function that builds the strategy for a given population size,
and `restart` returns the rebuilt strategy together with its params and state:

```python
import jax
from evosax.algorithms import CMA_ES
from evosax.core.restart import IPOPRestart, IPOPRestartParams, cma_cond, spread_cond
from evosax.problems import BBOBProblem

key = jax.random.key(0)

problem = BBOBProblem(fn_name="rastrigin", num_dims=8, f_opt=0.0)
key, key_init, key_sample = jax.random.split(key, 3)
problem_state = problem.init(key_init)
solution = problem.sample(key_sample)


def make_cma_es(population_size):
    return CMA_ES(population_size=population_size, solution=solution)


es = make_cma_es(16)
params = es.default_params

restarter = IPOPRestart(
    make_cma_es,
    initial_population_size=16,
    stop_criteria=(spread_cond, cma_cond),
    restart_params=IPOPRestartParams(min_num_gens=50),
)

key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)
restart_state = restarter.init(es)

for _ in range(500):
    key, key_ask, key_eval, key_tell, key_restart = jax.random.split(key, 5)

    population, state = es.ask(key_ask, state, params)
    fitness, problem_state, _ = problem.eval(key_eval, population, problem_state)
    state, metrics = es.tell(key_tell, population, fitness, state, params)

    if restarter.should_restart(population, fitness, state, params, restart_state):
        es, params, state, restart_state = restarter.restart(
            key_restart, es, state, params, restart_state
        )

print(restart_state.restart_counter, es.population_size, state.best_fitness)
```

`SimpleRestart` keeps the population size, so its `restart` only returns the new state
and restart state:

```python
from evosax.core.restart import SimpleRestart

restarter = SimpleRestart(stop_criteria=(spread_cond,))
restart_state = restarter.init()

if restarter.should_restart(population, fitness, state, params, restart_state):
    state, restart_state = restarter.restart(key, es, state, params, restart_state)
```

## Stop criteria

A restart is triggered when any of the given criteria is met and the run has completed
at least `min_num_gens` generations.

| Criterion | Met when |
| --- | --- |
| `spread_cond` | The difference between the best and worst fitness falls below `min_fitness_spread`. |
| `fitness_std_cond` | The standard deviation of the fitness falls below a tolerance. |
| `generation_cond` | The run reaches `generation_threshold` generations. |
| `cma_cond` | A CMA-ES specific condition on step size and covariance is met. |
| `amalgam_cond` | The iAMaLGaM distribution multiplier collapses. |

Thresholds are set through the restart params, for example
`IPOPRestartParams(min_num_gens=50, min_fitness_spread=1e-8)`.

!!! note "Where a restart starts"

    By default a restarted run starts from the `solution` the strategy was constructed
    with, so pass a meaningful starting point there rather than a placeholder. With
    `copy_mean=True` it starts from the mean of the previous run instead.

See the [API reference](../reference/core/restart.md) and the
[restart notebook](../examples.md) for more.
