# Examples

The repository contains a set of notebooks that can be run locally or in Google Colab.
Running them locally requires the `examples` extra, see
[Installation](getting-started/installation.md).

| Notebook | Topic | |
| --- | --- | --- |
| [Getting started](https://github.com/RobertTLange/evosax/blob/main/examples/00_getting_started.ipynb) | The ask-eval-tell API, fitness shaping and custom metrics. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/00_getting_started.ipynb) |
| [Black-box optimization benchmark](https://github.com/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb) | Visualizing BBOB functions and comparing strategies on them. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/01_bbob.ipynb) |
| [Reinforcement learning](https://github.com/RobertTLange/evosax/blob/main/examples/02_rl.ipynb) | Evolving MLP control policies in Gymnax and Brax. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/02_rl.ipynb) |
| [Vision](https://github.com/RobertTLange/evosax/blob/main/examples/03_vision.ipynb) | Training CNN classifiers on MNIST and CIFAR-10. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/03_vision.ipynb) |
| [Restart strategies](https://github.com/RobertTLange/evosax/blob/main/examples/04_restart_es.ipynb) | Simple, IPOP and BIPOP restart controllers. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/04_restart_es.ipynb) |
| [Diffusion Evolution](https://github.com/RobertTLange/evosax/blob/main/examples/05_diffusion_evolution.ipynb) | Optimization with Diffusion Evolution. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/05_diffusion_evolution.ipynb) |
| [Stein Variational ES](https://github.com/RobertTLange/evosax/blob/main/examples/06_sv_es.ipynb) | Stein Variational strategies on BBOB problems. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/06_sv_es.ipynb) |
| [Persistent and Noise-Reuse ES](https://github.com/RobertTLange/evosax/blob/main/examples/07_persistent_es.ipynb) | Evolution strategies for meta-learning problems. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/07_persistent_es.ipynb) |
| [Parallelization](https://github.com/RobertTLange/evosax/blob/main/examples/08_parallelization.ipynb) | Running a strategy across multiple devices. | [Open in Colab](https://colab.research.google.com/github/RobertTLange/evosax/blob/main/examples/08_parallelization.ipynb) |

## Scripts

Two standalone scripts evaluate the learned algorithms:

- [`learned_sphere.py`](https://github.com/RobertTLange/evosax/blob/main/examples/learned_sphere.py) runs the pretrained Learned ES, EvoTF
  and Learned GA on a two-dimensional sphere function.
- [`cma_anchored_les_brax.py`](https://github.com/RobertTLange/evosax/blob/main/examples/cma_anchored_les_brax.py) evaluates
  `CMAAnchoredLES` on eight Brax tasks.
