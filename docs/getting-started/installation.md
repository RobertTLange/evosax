# Installation

evosax requires Python 3.10 or later and a working
[JAX installation](https://jax.readthedocs.io/en/latest/installation.html). Install JAX
first if you need GPU or TPU support, for example on an NVIDIA GPU:

```bash
pip install -U "jax[cuda]"
```

Then install evosax from PyPI:

```bash
pip install evosax
```

To install the latest development version:

```bash
pip install git+https://github.com/RobertTLange/evosax.git@main
```

## Optional dependencies

The reinforcement learning and vision [problems](../guides/problems.md) and the example
notebooks need additional packages (Gymnax, Brax, PyTorch and torchvision). Install them
with the `examples` extra:

```bash
pip install "evosax[examples]"
```

[Optax](https://optax.readthedocs.io), which provides the optimizers and schedules
accepted by many algorithms, is installed together with evosax.
