# JAX-Based Evolution Strategies
[![Pyversions](https://img.shields.io/pypi/pyversions/evosax.svg?style=flat-square)](https://pypi.python.org/pypi/evosax) [![PyPI version](https://badge.fury.io/py/evosax.svg)](https://badge.fury.io/py/evosax)
<a href="docs/evosax_transparent_2.png"><img src="docs/evosax_transparent_2.png" width="200" align="right" /></a>

Tired of having to handle asynchronous processes for neuroevolution? Do you want to leverage massive vectorization and high-throughput accelerators for evolution strategies (ES)? `evosax` allows you to leverage JAX, XLA compilation and autovectorization to scale ES to your favorite accelerators. The API follows the classical `ask`, `evaluate`, `tell` cycle of ES and only requires you to `vmap` and `pmap` over the fitness function axes of choice. It includes popular strategies such as Simple Gaussian, CMA-ES, and different NES variants.

## Basic API Usage

```python
import jax
from evosax import CMA_ES
from evosax.problems import batch_rosenbrock

rng = jax.random.PRNGKey(0)
strategy = CMA_ES(popsize=20, num_dims=2, elite_ratio=0.5)
params = strategy.default_params
state = strategy.initialize(rng, params)

for t in range(num_generations):
    rng, rng_gen = jax.random.split(rng)
    x, state = strategy.ask(rng_gen, state, params)
    fitness = batch_rosenbrock(x, 1, 100)
    state = strategy.tell(x, fitness, state, params)

state["best_member"], state["best_fitness"]
```


# Implemented Evolution Strategies

| Strategy | Reference | Import | Example |
| --- | --- | ---  | --- |
| Simple Gaussian | :question: | [`Simple_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/simple_es.py) | [Low Dim. optimisation](https://github.com/RobertTLange/evosax/tree/main/examples/01_gaussian_strategy.ipynb)
| Simple Genetic | [Such et al. (2017)](https://arxiv.org/abs/1712.06567) | [`Simple_GA`](https://github.com/RobertTLange/evosax/tree/main/strategies/simple_ga.py) | [Low Dim. optimisation](https://github.com/RobertTLange/evosax/tree/main/examples/01_gaussian_strategy.ipynb)
| CMA-ES | [Hansen (2016)](https://arxiv.org/abs/1604.00772) | [`CMA_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/cma_es.py) | [Pendulum RL task](https://github.com/RobertTLange/evosax/tree/main/examples/pendulum_cma_es.ipynb)
| OpenAI-ES | [Salimans et al. (2017)](https://arxiv.org/pdf/1703.03864.pdf) | [`Open_NES`](https://github.com/RobertTLange/evosax/tree/main/strategies/open_nes.py) | [Simple Quadratic](https://github.com/RobertTLange/evosax/tree/main/examples/quadratic_open_nes.ipynb)
| PEPG | [Sehnke et al. (2009)](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=A64D1AE8313A364B814998E9E245B40A?doi=10.1.1.180.7104&rep=rep1&type=pdf) | [`PEPG_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/pepg_es.py)  | -
| Differential ES | [Storn & Price (1997)](https://www.metabolic-economics.de/pages/seminar_theoretische_biologie_2007/literatur/schaber/Storn1997JGlobOpt11.pdf) | [`Differential_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/differential_es.py)  | -
| Particle Swarm Optimization | [Kennedy & Eberhart (1995)](https://ieeexplore.ieee.org/document/488968) | [`PSO_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/pso_es.py)  | -
| Population-Based Training | [Jaderberg et al. (2017)](https://arxiv.org/abs/1711.09846) | [`PBT_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/pbt_es.py)  | -
| Persistent ES | [Vicol et al. (2021)](http://proceedings.mlr.press/v139/vicol21a.html) | [`Persistent_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/persistent_es.py)  | -
| x-NES | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) | `xNES` | -

## To Be Completed
| Strategy | Reference | Import | Example |
| --- | --- | ---  | --- |
| IPOP/BIPOP/SEP | - | :station:  | -
| NSLC | [Lehman & Stanley (2011)](https://direct.mit.edu/evco/article-abstract/19/2/189/1365/Abandoning-Objectives-Evolution-Through-the-Search) | :station:  | -
| MAP-Elites | [Mouret & Clune (2015)](https://arxiv.org/abs/1504.04909) | :station:  | -
| CMA-ME | [Fontaine et al. (2020)](https://arxiv.org/abs/1912.02400) | :station:  | -


## Vectorization Across Populations


## Vectorization Across Tasks


## Installation

`evosax` can directly be installed from PyPi.

```
pip install evosax
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

## Examples
* :book: [Blog post](https://roberttlange.github.io/posts/2020/12/neuroevolution-in-jax/): Walk through of CMA-ES and how to leverage JAX's primitives
* :notebook: [Low-dim. Optimisation](examples/01_gaussian_low_d.ipynb): Simple Gaussian strategy on Rosenbrock function
* :notebook: [MLP-Pendulum-Control](examples/02_cma_es_control.ipynb): CMA-ES on the `Pendulum-v0` gym task.
* :notebook: [CNN-MNIST-Classifier](examples/03_nes_cnn.ipynb): Open AI NES on MNIST-CNN.
* :notebook: [RNN-Meta-Bandit](examples/03_nes_cnn.ipynb): CMA-ES on an LSTM evolved to learn on bandit tasks.


## Contributing & Development

Feel free to ping me ([@RobertTLange](https://twitter.com/RobertTLange)), open an issue or start contributing yourself.
