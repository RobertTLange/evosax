from mle_toolbox.utils import set_random_seeds, get_configs_ready
import torch
import os, time
import jax, jaxlib
import jax.numpy as jnp
import numpy as np
from cma_es.cma_es_jax import init_cma_es, ask_cma_es, tell_cma_es
from ffw_pendulum import generation_rollout, flat_to_network


def speed_test_accelerator(num_evaluations, population_size, hidden_size,
                           net_config, train_config, log_config):
    """ Evaluate speed for different population sizes run. """
    # Want to eval gains from acc over different population size + archs
    train_config.pop_size = population_size
    train_config.hidden_size = hidden_size

    # Start by setting the random seeds for reproducibility
    rng = set_random_seeds(train_config.seed_id, return_key=True)

    # Define logger, CMA-ES strategy
    net_config.network_size[1] = int(train_config.hidden_size)
    num_params = (net_config.network_size[0] * net_config.network_size[1]
                  + net_config.network_size[1]
                  + net_config.network_size[1]*net_config.network_size[2]
                  + net_config.network_size[2])
    mean_init = jnp.zeros(num_params)
    elite_size = int(train_config.pop_size * train_config.elite_percentage)

    generation_times = []
    for eval in range(num_evaluations):
        es_params, es_memory = init_cma_es(mean_init,
                                           train_config.sigma_init,
                                           train_config.pop_size,
                                           elite_size)
        # Only track time for actual generation ask-tell inference
        start_t = time.time()

        # Train the network using the training loop
        run_single_generation(rng, elite_size,
                              train_config.num_evals_per_gen,
                              train_config.num_env_steps,
                              net_config.network_size,
                              dict(train_config.env_params),
                              es_params, es_memory)

        # Save wall-clock time for evaluation
        if eval > 0:
            generation_times.append(time.time() - start_t)
        else:
            jit_time = time.time() - start_t
    return np.mean(generation_times), np.std(generation_times), jit_time


def run_single_generation(rng, elite_size, num_evals_per_gen,
                          num_env_steps, network_size, env_params, es_params,
                          es_memory):
    """ Run the training loop over a set of epochs. """
    # Loop over different generations and search!
    rng, rng_input = jax.random.split(rng)
    x, es_memory = ask_cma_es(rng_input, es_memory, es_params)
    generation_params = flat_to_network(x, sizes=network_size)

    # Evaluate the fitness of the generation members
    rng, rng_input = jax.random.split(rng)
    rollout_keys = jax.random.split(rng_input, num_evals_per_gen)
    population_returns = generation_rollout(rollout_keys,
                                            generation_params,
                                            env_params, num_env_steps)
    values = - population_returns.mean(axis=1)

    # Update the CMA-ES strategy
    es_memory = tell_cma_es(x, values, elite_size, es_params, es_memory)
    return


if __name__ == "__main__":
    devices = jax.devices()
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

    print(f"JAX device: {devices}, {save_fname}")
    train_config, net_config, log_config = get_configs_ready(
        default_config_fname="configs/train/cma_config.json")

    num_evaluations = 100 + 1  # Dont use first - used for compilation
    population_sizes = [100, 250, 500, 750, 1000]
    network_sizes = [16, 48, 80, 112, 144]
    store_times = np.zeros((3, len(population_sizes), len(network_sizes)))
    for i, hidden_size in enumerate(network_sizes):
        for j, pop_size in enumerate(population_sizes):
            mean, std, jit_t = speed_test_accelerator(num_evaluations, pop_size, hidden_size,
                                                      net_config, train_config, log_config)
            # Jitted time, mean, std
            store_times[0, len(network_sizes)-1-i, j] = jit_t
            store_times[1, len(network_sizes)-1-i, j] = mean
            store_times[2, len(network_sizes)-1-i, j] = std
            print(hidden_size, pop_size, mean, std, jit_t)
    np.save(save_fname, store_times)
