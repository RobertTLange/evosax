# evosax

Evolution strategies in JAX.

evosax implements more than 30 evolution strategies behind a single `ask`-`eval`-`tell`
API, from classics such as CMA-ES and Differential Evolution to recent methods such as
OpenAI-ES and Diffusion Evolution. Every algorithm is a pure function of an explicit
state, so it works with `jax.jit`, `jax.vmap` and `jax.lax.scan` and runs on CPUs, GPUs
and TPUs without any distributed infrastructure.

```bash
pip install evosax
```

## A first run

```python
import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES


def sphere(x):
    return jnp.sum(x**2)


# An example solution tells the algorithm what the search space looks like
solution = jnp.zeros(8)
es = CMA_ES(population_size=32, solution=solution)
params = es.default_params

key = jax.random.key(0)
key, subkey = jax.random.split(key)
state = es.init(subkey, jnp.ones(8), params)

for _ in range(100):
    key, key_ask, key_tell = jax.random.split(key, 3)
    population, state = es.ask(key_ask, state, params)  # ask
    fitness = jax.vmap(sphere)(population)  # eval
    state, metrics = es.tell(key_tell, population, fitness, state, params)  # tell

print(state.best_fitness)  # close to 0
```

evosax minimizes. To maximize an objective, pass its negative to `tell`.

## Where to go next

- [Quickstart](getting-started/quickstart.md) walks through the loop above, compiles it
  with `jax.lax.scan` and optimizes a PyTree of parameters.
- [Concepts](guides/concepts.md) explains state, params and the two algorithm families.
- [Algorithms](algorithms.md) lists every implemented algorithm with its reference.
- [API reference](reference/algorithms/index.md) documents all classes and functions.

## Citing evosax

If you use evosax in your research, please cite the
[paper](https://arxiv.org/abs/2212.04180):

```bibtex
@article{evosax2022github,
    author  = {Robert Tjarko Lange},
    title   = {evosax: JAX-based Evolution Strategies},
    journal = {arXiv preprint arXiv:2212.04180},
    year    = {2022},
}
```
