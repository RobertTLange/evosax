import copy, commentjson
import torch
import os, time
import jax, jaxlib
import jax.numpy as jnp
import numpy as np
from dotmap import DotMap
from evosax.strategies.cma_es import init_strategy, ask, tell
from evosax.utils import flat_to_mlp

import sys

sys.path.append("..")
from examples.ffw_pendulum import (
    generation_rollout,
    generation_rollout_no_jit,
    generation_rollout_pmap,
)


def load_config(config_fname: str):
    """Load in a config JSON file and return as a dictionary"""
    json_config = commentjson.loads(open(config_fname, "r").read())
    dict_config = DotMap(json_config)

    # Make inner dictionaries indexable like a class
    for key, value in dict_config.items():
        if isinstance(value, dict):
            dict_config[key] = DotMap(value)
    return dict_config


def benchmark_accelerator(
    num_evaluations,
    population_size,
    hidden_size,
    net_config,
    train_config,
    log_config,
    use_jit=True,
    use_tpu=False,
):
    """Evaluate speed for different population sizes run."""
    # Want to eval gains from acc over different population size + archs
    train_config.pop_size = population_size
    train_config.hidden_size = hidden_size

    # Start by setting the random seeds for reproducibility
    rng = jax.random.PRNGKey(train_config.seed_id)

    # Define logger, CMA-ES strategy
    net_config.network_size[1] = int(train_config.hidden_size)
    num_params = (
        net_config.network_size[0] * net_config.network_size[1]
        + net_config.network_size[1]
        + net_config.network_size[1] * net_config.network_size[2]
        + net_config.network_size[2]
    )
    mean_init = jnp.zeros(num_params)
    elite_size = int(train_config.pop_size * train_config.elite_percentage)

    generation_times = []
    for eval in range(num_evaluations):
        es_params, es_memory = init_strategy(
            mean_init, train_config.sigma_init, train_config.pop_size, elite_size
        )
        # Only track time for actual generation ask-tell inference
        start_t = time.time()

        # Train the network using the training loop
        run_single_generation(
            rng,
            elite_size,
            train_config.num_evals_per_gen,
            train_config.num_env_steps,
            net_config.network_size,
            dict(train_config.env_params),
            es_params,
            es_memory,
            use_jit,
            use_tpu,
        )

        # Save wall-clock time for evaluation
        if eval > 0:
            generation_times.append(time.time() - start_t)
        else:
            jit_time = time.time() - start_t
    return np.mean(generation_times), np.std(generation_times), jit_time


def run_single_generation(
    rng,
    elite_size,
    num_evals_per_gen,
    num_env_steps,
    network_size,
    env_params,
    es_params,
    es_memory,
    use_jit=True,
    use_tpu=False,
):
    """Run the training loop over a set of epochs."""
    # Loop over different generations and search!
    rng, rng_input = jax.random.split(rng)
    x, es_memory = ask(rng_input, es_params, es_memory)
    generation_params = flat_to_mlp(x, sizes=network_size)

    # Evaluate the fitness of the generation members
    rng, rng_input = jax.random.split(rng)
    rollout_keys = jax.random.split(rng_input, num_evals_per_gen)

    if use_jit:
        population_returns = generation_rollout(
            rollout_keys, generation_params, env_params, num_env_steps
        )
    elif use_tpu:
        population_returns = generation_rollout_pmap(
            rollout_keys, generation_params, env_params, num_env_steps
        )
    else:
        population_returns = generation_rollout_no_jit(
            rollout_keys, generation_params, env_params, num_env_steps
        )

    values = -population_returns.mean(axis=1)

    # Update the CMA-ES strategy
    es_memory = tell(x, values, es_params, es_memory)
    return


if __name__ == "__main__":
    devices = jax.devices()
    use_jit = False
    if type(devices[0]) == jaxlib.xla_extension.CpuDevice:
        save_fname = "cpu_speed"
    elif type(devices[0]) == jaxlib.xla_extension.GpuDevice:
        if torch.cuda.get_device_name(0) == "Tesla K40c":
            save_fname = "gpu_tesla_k40_speed"
        elif torch.cuda.get_device_name(0) == "GeForce RTX 2080 Ti":
            save_fname = "gpu_rtx2080ti_speed"
        else:
            save_fname = "gpu_speed"
    elif type(devices[0]) == jaxlib.xla_extension.TpuDevice:
        save_fname = "tpu_speed"

    if use_jit:
        save_fname = "results/" + save_fname + "_jit"
    else:
        save_fname = "results/" + save_fname + "_no_jit"

    print(f"JAX device: {devices}, {save_fname}")
    config = load_config("../examples/configs/train/cma_config.json")
    train_config, net_config, log_config = (
        config.train_config,
        config.net_config,
        config.log_config,
    )

    num_evaluations = 1000 + 1  # Dont use first - used for compilation
    population_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # network_sizes = [16, 48, 80, 112, 144]
    network_sizes = [48]
    store_times = np.zeros((3, len(network_sizes), len(population_sizes)))
    for i, hidden_size in enumerate(network_sizes):
        for j, pop_size in enumerate(population_sizes):
            mean, std, jit_t = benchmark_accelerator(
                num_evaluations,
                pop_size,
                hidden_size,
                net_config,
                train_config,
                log_config,
                use_jit,
            )
            # Jitted time, mean, std
            store_times[0, len(network_sizes) - 1 - i, j] = jit_t
            store_times[1, len(network_sizes) - 1 - i, j] = mean
            store_times[2, len(network_sizes) - 1 - i, j] = std
            print(hidden_size, pop_size, mean, std, jit_t)
    np.save(save_fname, store_times)
