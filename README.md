# JAX-Based Evolutionary Strategies
[![Pyversions](https://img.shields.io/pypi/pyversions/evosax.svg?style=flat-square)](https://pypi.python.org/pypi/evosax) [![PyPI version](https://badge.fury.io/py/evosax.svg)](https://badge.fury.io/py/evosax)
<a href="docs/evosax_transparent_2.png"><img src="docs/evosax_transparent_2.png" width="200" align="right" /></a>

Tired of having to handle asynchronous processes for neuroevolution? Do you want to leverage massive vectorization and high-throughput accelerators for evolutionary strategies (ES)? `evosax` allows you to leverage JAX, XLA compilation and autovectorization to scale ES to your favorite accelerators. The API follows the classical `ask`, `evaluate`, `tell` cycle of ES and only requires you to `vmap` and `pmap` over the fitness function axes of choice. It includes popular strategies such as Simple Gaussian, CMA-ES, and different NES variants.

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


# Implemented Evolutionary Strategies

| Strategy | Reference | Implemented | Source Code | Example |
| --- | --- | --- | --- | --- |
| Simple Gaussian | :question: | :heavy_check_mark:  | [Click](evosax/strategies/gaussian.py) | [Low Dim. optimisation](notebooks/01_gaussian_strategy.ipynb)
| CMA-ES | [Hansen (2016)](https://arxiv.org/abs/1604.00772) | :heavy_check_mark:  | [Click](evosax/strategies/cma_es.py) | [Pendulum RL task](notebooks/pendulum_cma_es.ipynb)
| OpenAI-ES | [Salimans et al. (2017)](https://arxiv.org/pdf/1703.03864.pdf) | :heavy_check_mark:  | [Click](evosax/strategies/open_nes.py) | [Simple Quadratic](notebooks/quadratic_open_nes.ipynb)
| IPOP/BIPOP/SEP | - | :station:  | - | -
| NES | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) | :station:  | - | -
| PEPG | [Sehnke et al. (2009)](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=A64D1AE8313A364B814998E9E245B40A?doi=10.1.1.180.7104&rep=rep1&type=pdf) | :station:  | - | -



## Vectorization Across Populations


## Vectorization Across Tasks


## Installation

`evosax` can directly be installed from PyPi.

```
pip install evosax
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

## Intro, Examples, Notebooks & Colabs
* :book: [Blog post](https://roberttlange.github.io/posts/2020/12/neuroevolution-in-jax/): Walk through of CMA-ES and how to leverage JAX's primitives
* :notebook: [Low-dim. Optimisation](notebooks/01_gaussian_low_d.ipynb): Simple Gaussian strategy on Rosenbrock function
* :notebook: [MLP-Pendulum-Control](notebooks/02_cma_es_control.ipynb): CMA-ES on the `Pendulum-v0` gym task.
* :notebook: [CNN-MNIST-Classifier](notebooks/03_nes_cnn.ipynb): Open AI NES on MNIST-CNN.
* :notebook: [RNN-Meta-Bandit](notebooks/03_nes_cnn.ipynb): CMA-ES on an LSTM evolved to learn on bandit tasks.


## Contributing & Development

Feel free to ping me ([@RobertTLange](https://twitter.com/RobertTLange)), open an issue or start contributing yourself.
