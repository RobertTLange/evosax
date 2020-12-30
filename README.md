# `evosax` - Evolutionary Strategies in JAX

Are you tired of having to handle asynchronous processes for neuroevolution? Do you want to leverage massive vectorization for high-throughput accelerators for evolutionary strategies (ES)? `evosax` allows you to leverage JAX and to scale ES to GPUs/TPUs using XLA compilation.

## Basic API Usage

```python
from evosax.strategies.cma_es import init_strategy, ask, tell

# Initialize the CMA evolutionary strategy
params, memory = init_strategy(mean, sigma, pop_size, elite_size)

# Loop over number of generations using ask-eval-tell API
for g in range(num_generations):
    # Explicitly handle random number generation
    rng, rng_input = jax.random.split(rng)

    # Ask for the next generation population to test
    x, memory = ask(rng_input, memory, params)

    # Evaluate the fitness of the generation members
    fitness = evaluate_fitness(x)

    # Tell/Update the CMA-ES with newest data points
    memory = tell(x, values, elite_size, params, memory)
```

<details><summary>
Implemented evolutionary strategies.

</summary>

| Strategy | Reference | Implemented | Source Code | Example |
| --- | --- | --- | --- | --- |
| Classic Control | `Pendulum-v0` | :heavy_check_mark:  | :heavy_check_mark: |
</details>


## Installing `evosax` and dependencies

`evosax` can be directly installed from PyPi.

```
pip install evosax
```

Alternatively, you can clone this repository and afterwards 'manually' install the toolbox (preferably in a clean Python 3.6 environment):

```
git clone https://github.com/RobertTLange/evosax.git
cd evosax
pip install -e .
```

This will install all required dependencies. Note that by default the `evosax` installation will install CPU-only `jaxlib`. In order to install the CUDA-supported version, simply upgrade to the right `jaxlib`. E.g. for a CUDA 10.1 driver:

```
pip install --upgrade jaxlib==0.1.57+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

You can find more details in the [JAX documentation](https://github.com/google/jax#installation). Finally, please note that `evosax` is only tested for Python 3.6. You can directly run the test from the repo directory via `pytest`.

## Benchmarking Details

Throughout our benchmarks we make use of the [`gymnax`](https://github.com/RobertTLange/gymnax) package for accelerated RL environments. Thereby, we can `jit` through entire RL episode rollouts.

<details> <summary>
  Device and benchmark details.

</summary>

| Name | Framework | Description | Device | Steps in Ep. | Number of Ep. |
| --- | --- | --- | --- | --- | --- |
CPU-STEP-GYM | OpenAI gym/NumPy | Single transition |2,7 GHz Intel Core i7| 1 | - |
</details>


The speed comparisons were benchmarked for the devices and transition rollout settings listed above. Stochastic fitness evaluations are collected synchronously and using a composition of `jit`, `vmap`/`pmap` (over evaluations and population members) and `lax.scan` (over sequential fitness evaluations).


## Intro, Examples, Notebooks & Colabs
* :book: [Blog post](https://roberttlange.github.io/posts/2020/12/neuroevolution-in-jax/): Walk through of CMA-ES and how to leverage JAX in ES.
* :notebook: [Low-D Optimization](examples/optimisation_cma_es.ipynb): CMA-ES on Rosenbrock, Himmelblau, 6-Hump Camel Fct.
* :notebook: [MLP-Control](examples/pendulum_cma_es.ipynb): CMA-ES on the `Pendulum-v0` gym task.


## Contributing & Development

Feel free to ping me ([@RobertTLange](https://twitter.com/RobertTLange)), open an issue or start contributing yourself.

## TODOs, Notes & Questions
- [ ] Benchmark plots
- [ ] Clean up visualizations/proper API
- [ ] Implement more strategies
    - [ ] Add restarts for CMA-ES
    - [ ] Add simple Gaussian strategy
- [ ] Implement more examples
    - [ ] MNIST classification example - MLP/CNNs
    - [ ] Small RNN example
    - [ ] Use flax/haiku as NN library
- [ ] Add different param -> network reshaping helpers
- [ ] Connect notebooks with example Colab https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk
