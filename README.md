# `evosax`: JAX-Based Evolution Strategies
[![Pyversions](https://img.shields.io/pypi/pyversions/evosax.svg?style=flat-square)](https://pypi.python.org/pypi/evosax) [![PyPI version](https://badge.fury.io/py/evosax.svg)](https://badge.fury.io/py/evosax)[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/getting_started.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/RobertTLange/evosax/branch/main/graph/badge.svg?token=5FUSX35KWO)](https://codecov.io/gh/RobertTLange/evosax)
<a href="https://github.com/RobertTLange/evosax/blob/main/docs/evosax_transparent_2.png?raw=true"><img src="https://github.com/RobertTLange/evosax/blob/main/docs/evosax_transparent_2.png?raw=true" width="170" align="right" /></a>

Tired of having to handle asynchronous processes for neuroevolution? Do you want to leverage massive vectorization and high-throughput accelerators for evolution strategies (ES)? `evosax` allows you to leverage JAX, XLA compilation and autovectorization to scale ES to your favorite accelerators. The API follows the classical `ask`, `evaluate`, `tell` cycle of ES and only requires you to `vmap` and `pmap` over the fitness function axes of choice. It includes popular strategies such as Simple Gaussian, CMA-ES, and different NES variants.


## Basic `evosax` API Usage 🍲

```python
import jax
from evosax import CMA_ES
from evosax.problems import batch_rosenbrock

# Instantiate the search strategy
rng = jax.random.PRNGKey(0)
strategy = CMA_ES(popsize=20, num_dims=2, elite_ratio=0.5)
params = strategy.default_params
state = strategy.initialize(rng, params)

# Run the ask-eval-tell loop
for t in range(num_generations):
    rng, rng_gen = jax.random.split(rng)
    x, state = strategy.ask(rng_gen, state, params)
    fitness = batch_rosenbrock(x, 1, 100)
    state = strategy.tell(x, fitness, state, params)

# Get best overall population member & its fitness
state["best_member"], state["best_fitness"]
```


## Implemented Evolution Strategies 🦎

| Strategy | Reference | Import | Example |
| --- | --- | ---  | --- |
| CMA-ES | [Hansen (2016)](https://arxiv.org/abs/1604.00772) | [`CMA_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/strategies/cma_es.py) | [Pendulum RL task](https://github.com/RobertTLange/evosax/tree/main/examples/pendulum_cma_es.ipynb)
| Differential ES | [Storn & Price (1997)](https://www.metabolic-economics.de/pages/seminar_theoretische_biologie_2007/literatur/schaber/Storn1997JGlobOpt11.pdf) | [`Differential_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/strategies/differential_es.py)  | -
| OpenAI-ES | [Salimans et al. (2017)](https://arxiv.org/pdf/1703.03864.pdf) | [`Open_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/open_es.py) | [Simple Quadratic](https://github.com/RobertTLange/evosax/tree/main/examples/quadratic_open_nes.ipynb)
| Particle Swarm Optimization | [Kennedy & Eberhart (1995)](https://ieeexplore.ieee.org/document/488968) | [`PSO_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/strategies/pso_es.py)  | -
| PEPG | [Sehnke et al. (2010)](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=A64D1AE8313A364B814998E9E245B40A?doi=10.1.1.180.7104&rep=rep1&type=pdf) | [`PEPG_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/strategies/pepg_es.py)  | -
| ARS | [Mania et al. (2018)](https://arxiv.org/pdf/1803.07055.pdf) | [`Augmented_RS`](https://github.com/RobertTLange/evosax/tree/main/evosax/strategies/ars.py)  | -
| Persistent ES | [Vicol et al. (2021)](http://proceedings.mlr.press/v139/vicol21a.html) | [`Persistent_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/persistent_es.py)  | -
| Population-Based Training | [Jaderberg et al. (2017)](https://arxiv.org/abs/1711.09846) | [`PBT_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/strategies/pbt_es.py)  | -
| Simple Gaussian | ❓ | [`Simple_ES`](https://github.com/RobertTLange/evosax/tree/main/strategies/evosax/simple_es.py) | [Low Dim. optimisation](https://github.com/RobertTLange/evosax/tree/main/examples/01_gaussian_strategy.ipynb)
| Simple Genetic | [Such et al. (2017)](https://arxiv.org/abs/1712.06567) | [`Simple_GA`](https://github.com/RobertTLange/evosax/tree/main/evosax/strategies/simple_ga.py) | [Low Dim. optimisation](https://github.com/RobertTLange/evosax/tree/main/examples/01_gaussian_strategy.ipynb)
| x-NES | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) | [`xNES`](https://github.com/RobertTLange/evosax/tree/main/evosax/strategies/xnes.py)  | -

## Installation ⏳

`evosax` can directly be installed from PyPi.

```
pip install evosax
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

## Examples 📖
* 📖 [Blog post](https://roberttlange.github.io/posts/2020/12/neuroevolution-in-jax/): Walk through of CMA-ES and how to leverage JAX's primitives
* 📓 [Low-dim. Optimisation](https://github.com/RobertTLange/evosax/blob/main/examples/01_gaussian_low_d.ipynb): Simple Gaussian strategy on Rosenbrock function
* 📓 [MLP-Pendulum-Control](https://github.com/RobertTLange/evosax/blob/main/examples/02_cma_es_control.ipynb): CMA-ES on the `Pendulum-v0` gym task.
* 📓 [CNN-MNIST-Classifier](https://github.com/RobertTLange/evosax/blob/main/examples/03_nes_cnn.ipynb): Open AI NES on MNIST-CNN.
* 📓 [RNN-Meta-Bandit](https://github.com/RobertTLange/evosax/blob/main/examples/03_nes_cnn.ipynb): CMA-ES on an LSTM evolved to learn on bandit tasks.


## Contributing & Development 🧑‍🤝‍🧑

Feel free to ping me ([@RobertTLange](https://twitter.com/RobertTLange)), open an issue or start contributing yourself.

<!-- ## To Be Completed
| Strategy | Reference | Import | Example |
| --- | --- | ---  | --- |
| IPOP/BIPOP/SEP | - | 🚉  | -
| NSLC | [Lehman & Stanley (2011)](https://direct.mit.edu/evco/article-abstract/19/2/189/1365/Abandoning-Objectives-Evolution-Through-the-Search) | 🚉 | -
| MAP-Elites | [Mouret & Clune (2015)](https://arxiv.org/abs/1504.04909) |🚉  | -
| CMA-ME | [Fontaine et al. (2020)](https://arxiv.org/abs/1912.02400) | 🚉  | - -->
