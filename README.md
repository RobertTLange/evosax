# `evosax`: Evolution Strategies in JAX 🦎

[![Pyversions](https://img.shields.io/pypi/pyversions/evosax.svg?style=flat)](https://pypi.python.org/pypi/evosax) [![PyPI version](https://badge.fury.io/py/evosax.svg)](https://badge.fury.io/py/evosax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/RobertTLange/evosax/branch/main/graph/badge.svg?token=5FUSX35KWO)](https://codecov.io/gh/RobertTLange/evosax)
[![Paper](http://img.shields.io/badge/paper-arxiv.2212.04180-B31B1B.svg)](http://arxiv.org/abs/2212.04180)
<a href="https://github.com/RobertTLange/evosax/blob/main/docs/logo.png?raw=true"><img src="https://github.com/RobertTLange/evosax/blob/main/docs/logo.png?raw=true" width="170" align="right" /></a>

Tired of having to handle asynchronous processes for neuroevolution? Do you want to leverage massive vectorization and high-throughput accelerators for Evolution Strategies? `evosax` provides a comprehensive, high-performance library that implements Evolution Strategies (ES) in JAX. By leveraging XLA compilation and JAX's transformation primitives, `evosax` enables researchers and practitioners to efficiently scale evolutionary algorithms to modern hardware accelerators without the traditional overhead of distributed implementations.

The API follows the classical `ask`-`eval`-`tell` cycle of ES, with full support for JAX's transformations (`jit`, `vmap`, `lax.scan`). The library includes 30+ evolution strategies, from classics like CMA-ES and Differential Evolution to modern approaches like OpenAI-ES and Diffusion Evolution.

**Get started here** 👉 [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/00_getting_started.ipynb)

## Basic `evosax` API Usage 🍲

```python
import jax
from evosax.algorithms import CMA_ES


# Instantiate the search strategy
es = CMA_ES(population_size=32, solution=dummy_solution)
params = es.default_params

# Initialize state
key = jax.random.key(0)
state = es.init(key, params)

# Ask-Eval-Tell loop
for i in range(num_generations):
    key, key_ask, key_eval = jax.random.split(key, 3)

    # Generate a set of candidate solutions to evaluate
    population, state = es.ask(key_ask, state, params)

    # Evaluate the fitness of the population
    fitness = ...

    # Update the evolution strategy
    state = es.tell(population, fitness, state, params)

# Get best solution
state.best_solution, state.best_fitness
```

## Implemented Evolution Strategies 🦎

| Strategy                    | Reference                                                                                                                                                | Import                                                                                                   | Example |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------| --- |
| Simple Evolution Strategy   | [Rechenberg (1978)](https://link.springer.com/chapter/10.1007/978-3-642-81283-5_8)                                                                       | [`SimpleES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/simple_es.py)            | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| OpenAI-ES                   | [Salimans et al. (2017)](https://arxiv.org/abs/1703.03864)                                                                                           | [`Open_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/open_es.py)                | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/02_rl.ipynb)
| CMA-ES                      | [Hansen & Ostermeier (2001)](https://arxiv.org/abs/1604.00772)                                                             | [`CMA_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/cma_es.py)                 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Sep-CMA-ES                  | [Ros & Hansen (2008)](https://hal.inria.fr/inria-00287367/document)                                                                                      | [`Sep_CMA_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/sep_cma_es.py)         | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| xNES                        | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf)                                                               | [`xNES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/xnes.py)                     | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| SNES                        | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf)                                                               | [`SNES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/snes.py)                    | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| MA-ES                       | [Bayer & Sendhoff (2017)](https://ieeexplore.ieee.org/document/7875115)                                                                                     | [`MA_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/ma_es.py)                   | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| LM-MA-ES                    | [Loshchilov et al. (2017)](https://arxiv.org/pdf/1705.06693.pdf)                                                                                         | [`LM_MA_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/lm_ma_es.py)             | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Rm_ES                       | [Li & Zhang (2017)](https://ieeexplore.ieee.org/document/8080257)                                                                                        | [`Rm_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/rm_es.py)                    | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| PGPE                        | [Sehnke et al. (2010)](https://link.springer.com/chapter/10.1007/978-3-540-87536-9_40) | [`PGPE`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/pgpe.py)                     | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/02_rl.ipynb)
| ARS                         | [Mania et al. (2018)](https://arxiv.org/pdf/1803.07055)                                                                                              | [`ARS`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/ars.py)                       | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/02_rl.ipynb)
| ESMC                        | [Merchant et al. (2021)](https://arxiv.org/abs/2107.09661)                                                                            | [`ESMC`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/esmc.py)                     | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Persistent ES               | [Vicol et al. (2021)](https://arxiv.org/abs/2112.13835)                                                                                   | [`PersistentES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/persistent_es.py)    | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/07_persistent_es.ipynb)
| Noise-Reuse ES              | [Li et al. (2023)](https://arxiv.org/abs/2304.12180)                                                                                                 | [`NoiseReuseES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/noise_reuse_es.py)   | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/07_persistent_es.ipynb)
| CR-FM-NES                   | [Nomura & Ono (2022)](https://arxiv.org/abs/2201.11422)                                                                                                  | [`CR_FM_NES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/cr_fm_nes.py)           | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Guided ES                   | [Maheswaranathan et al. (2018)](https://arxiv.org/abs/1806.10230)                                                                                        | [`GuidedES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/guided_es.py)            | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| ASEBO                       | [Choromanski et al. (2019)](https://arxiv.org/abs/1903.04268)                                                                                            | [`ASEBO`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/asebo.py)                   | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Discovered ES               | [Lange et al. (2023a)](https://arxiv.org/abs/2211.11260)                                                                                                 | [`DiscoveredES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/discovered_es.py)                       | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| LES                  | [Lange et al. (2023a)](https://arxiv.org/abs/2211.11260)                                                                                                 | [`LearnedES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/learned_es.py)                       | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| EvoTF                       | [Lange et al. (2024)](https://arxiv.org/abs/2403.02985)                                                                                                  | [`EvoTF_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/evotf_es.py)             | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| iAMaLGaM-Full               | [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf)                                                                                                     | [`iAMaLGaM_Full`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/iamalgam_full.py)   |[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| iAMaLGaM-Univariate         | [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf)                                                                                                     | [`iAMaLGaM_Univariate`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/iamalgam_univariate.py) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Gradientless Descent        | [Golovin et al. (2019)](https://arxiv.org/abs/1911.06317)                                                                                            | [`GradientlessDescent`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/gradientless_descent.py)                       | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Simulated Annealing         | [Rasdi Rere et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1877050915035759)                                                          | [`SimAnneal`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/simulated_annealing.py)          | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Hill Climbing               | [Rasdi Rere et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1877050915035759)                                                          | [`HillClimbing`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/hill_climbing.py)          | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Random Search               | [Bergstra & Bengio (2012)](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)                                                             | [`RandomSearch`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/random.py)           | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| SV-CMA-ES                   | [Braun et al. (2024)](https://arxiv.org/abs/2410.10390)                                                                                                  | [`SV_CMA_ES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/sv/sv_cma_es.py)           | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/06_sv_es.ipynb)
| SV-OpenAI-ES                | [Liu et al. (2017)](https://arxiv.org/abs/2410.10390)                                                                                                    | [`SV_OpenES`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based/sv/sv_open_es.py)          | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/06_sv_es.ipynb)
| Simple Genetic Algorithm    | [Such et al. (2017)](https://arxiv.org/abs/1712.06567)                                                                                                   | [`SimpleGA`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/simple_ga.py)            | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| MR15-GA                     | [Rechenberg (1978)](https://link.springer.com/chapter/10.1007/978-3-642-81283-5_8)                                                                       | [`MR15_GA`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/mr15_ga.py)               | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| SAMR-GA                     | [Clune et al. (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000187)                                                    | [`SAMR_GA`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/samr_ga.py)               | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| GESMR-GA                    | [Kumar et al. (2022)](https://arxiv.org/abs/2204.04817)                                                                                                  | [`GESMR_GA`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/gesmr_ga.py)             | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| LGA                         | [Lange et al. (2023b)](https://arxiv.org/abs/2304.03995)                                                                                                 | [`LearnedGA`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/learned_ga.py)                       | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Diffusion Evolution         | [Zhang et al. (2024)](https://arxiv.org/pdf/2410.02543)                                                                                                  | [`DiffusionEvolution`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/diffusion_evolution.py)  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Differential Evolution      | [Storn & Price (1997)](https://link.springer.com/article/10.1023/A:1008202821328)            | [`DifferentialEvolution`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/differential_evolution.py)                         | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)
| Particle Swarm Optimization | [Kennedy & Eberhart (1995)](https://ieeexplore.ieee.org/document/488968)                                                                                 | [`PSO`](https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based/pso.py)                       | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb)

## Installation ⏳

You will need Python 3.10 or later, and a working JAX installation.

Then, install `evosax` from PyPi:

```bash
pip install evosax
```

To upgrade to the latest version of `evosax`, you can use:

```bash
pip install git+https://github.com/RobertTLange/evosax.git@main
```

## Examples 📖

* 📓 [Getting Started](https://github.com/RobertTLange/evosax/blob/main/examples/00_getting_started.ipynb) - Introduction to the library
* 📓 [Black Box Optimization Benchmark](https://github.com/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb) - Optimization of common test functions
* 📓 [Reinforcement Learning](https://github.com/RobertTLange/evosax/blob/main/examples/02_rl.ipynb) - Learning MLP control policies
* 📓 [Vision](https://github.com/RobertTLange/evosax/blob/main/examples/03_vision.ipynb) - Training CNNs for classification
* 📓 [Restart ES](https://github.com/RobertTLange/evosax/blob/main/examples/04_restart_es.ipynb) - Implementing restart strategies
* 📓 [Diffusion Evolution](https://github.com/RobertTLange/evosax/blob/main/examples/05_diffusion_evolution.ipynb) - Optimization with diffusion evolution
* 📓 [Stein Variational ES](https://github.com/RobertTLange/evosax/blob/main/examples/06_sv_es.ipynb) - Using SV-ES on BBOB problems
* 📓 [Persistent/Noise-Reuse ES](https://github.com/RobertTLange/evosax/blob/main/examples/07_persistent_es.ipynb) - ES for meta-learning problems
* 📓 [Parallelization](https://github.com/RobertTLange/evosax/blob/main/examples/08_parallelization.ipynb) - ES with parallelization on multiple devices

## Key Features 💎

- **Comprehensive Algorithm Collection**: 30+ classic and modern evolution strategies with a unified API
- **JAX Acceleration**: Fully compatible with JAX transformations for speed and scalability
- **Vectorization & Parallelization**: Fast execution on CPUs, GPUs, and TPUs
- **Production Ready**: Well-tested, documented, and used in research environments
- **Batteries Included**: Comes with optimizers like ClipUp, fitness shaping, and restart strategies

## Related Resources 📚

* 📺 [Rob's MLC Research Jam Talk](https://www.youtube.com/watch?v=Wn6Lq2bexlA&t=51s) - Overview at the ML Collective Research Jam
* 📝 [Rob's 02/2021 Blog](https://roberttlange.github.io/posts/2021/02/cma-es-jax/) - Blog post on implementing CMA-ES in JAX
* 💻 [Evojax](https://github.com/google/evojax) - Hardware-Accelerated Neuroevolution with great rollout wrappers.
* 💻 [QDax](https://github.com/adaptive-intelligent-robotics/QDax): Quality-Diversity algorithms in JAX.

## Citing `evosax` ✏️

If you use `evosax` in your research, please cite the following [paper](https://arxiv.org/abs/2212.04180):
```bibtex
@article{evosax2022github,
    author  = {Robert Tjarko Lange},
    title   = {evosax: JAX-based Evolution Strategies},
    journal = {arXiv preprint arXiv:2212.04180},
    year    = {2022},
}
```

## Acknowledgements 🙏

We acknowledge financial support by the [Google TRC](https://sites.research.google/trc/about/) and the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2002/1 ["Science of Intelligence"](https://www.scienceofintelligence.de/) - project number 390523135.

## Contributing 👷

Contributions are welcome! If you find a bug or are missing your favorite feature, please [open an issue](https://github.com/RobertTLange/evosax/issues) or submit a pull request following our [contribution guidelines](CONTRIBUTING.md) 🤗.

## Disclaimer ⚠️

This repository contains independent reimplementations of LES and DES based and is unrelated to Google DeepMind. The implementation has been tested to reproduce the official results on a range of tasks.
