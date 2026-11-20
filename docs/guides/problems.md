# Problems

evosax ships a few ready-made problems for benchmarking and for neuroevolution. They all
share one interface:

- `problem.init(key)` creates the problem state,
- `problem.sample(key)` draws a solution, which is handy as the example `solution` of an
  algorithm,
- `problem.eval(key, population, problem_state)` returns
  `(fitness, problem_state, info)`.

You do not need a problem class to use evosax. Any function that maps a population to
fitness values works, as shown in the [quickstart](../getting-started/quickstart.md).

## Black-box optimization benchmark

`BBOBProblem` implements the functions of the BBOB benchmark in any number of
dimensions. Lower fitness is better, so it can be passed to `tell` directly.

```python
import jax
from evosax.problems import BBOBProblem, bbob_fns

print(list(bbob_fns)[:3])  # names of the available functions

problem = BBOBProblem(fn_name="rastrigin", num_dims=8, f_opt=0.0)

key = jax.random.key(0)
key, key_init, key_sample = jax.random.split(key, 3)
problem_state = problem.init(key_init)
solution = problem.sample(key_sample)
```

`MetaBBOBProblem` samples whole families of BBOB tasks with random optima, rotations
and dimensions, which is useful for benchmarking across many tasks at once. Its
interface has one more step: `sample_params(key)` draws a task, and `init` and `eval`
take the resulting `params` as an additional last argument.

## Reinforcement learning

`GymnaxProblem` and `BraxProblem` evaluate a policy network by rolling it out in a
[Gymnax](https://github.com/RobertTLange/gymnax) or
[Brax](https://github.com/google/brax) environment. They require the `examples` extra,
see [Installation](../getting-started/installation.md).

```python
from evosax.algorithms import Open_ES
from evosax.problems import GymnaxProblem
from evosax.problems.networks import MLP, categorical_output_fn

policy = MLP(layer_sizes=(64, 64, 2), output_fn=categorical_output_fn)

problem = GymnaxProblem(
    env_name="CartPole-v1",
    policy=policy,
    episode_length=200,
    num_rollouts=16,
    use_normalize_obs=True,
)

key, key_init, key_sample = jax.random.split(key, 3)
problem_state = problem.init(key_init)
solution = problem.sample(key_sample)  # initial policy parameters

es = Open_ES(population_size=64, solution=solution)
params = es.default_params

key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)

key, key_ask, key_eval, key_tell = jax.random.split(key, 4)
population, state = es.ask(key_ask, state, params)
fitness, problem_state, info = problem.eval(key_eval, population, problem_state)
state, metrics = es.tell(key_tell, population, -fitness, state, params)
```

!!! note "Returns are maximized"

    The RL problems return the mean episode return, where higher is better. Negate it
    when calling `tell`.

## Vision

`TorchVisionProblem` trains a classifier on a torchvision dataset: `"MNIST"`,
`"FashionMNIST"`, `"CIFAR10"` or `"SVHN"`. It loads the data with PyTorch, so it also
requires the `examples` extra.

```python
from evosax.problems import CNN, TorchVisionProblem, identity_output_fn

network = CNN(
    num_filters=[8, 16],
    kernel_sizes=[(5, 5), (5, 5)],
    strides=[(1, 1), (1, 1)],
    mlp_layer_sizes=[10],
    output_fn=identity_output_fn,
)

problem = TorchVisionProblem(task_name="MNIST", network=network, batch_size=1024)
```

`eval` returns the classification loss on a training batch, which can be passed to
`tell` directly. The accuracy is available as `info["accuracy"]`.

## Writing your own

To package an objective as a reusable problem, subclass
[`Problem`](../reference/problems/index.md) and implement `eval` and `sample`.

The [example notebooks](../examples.md) show complete training loops for the
reinforcement learning and vision problems.
