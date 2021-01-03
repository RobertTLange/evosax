from mle_toolbox.utils import (set_random_seeds,
                               get_configs_ready,
                               DeepLogger)

import os
import jax
import jax.numpy as jnp
import numpy as np
from evosax.strategies.cma_es import init_strategy, ask, tell, check_termination
from evosax.utils import init_logger, update_logger, flat_to_mlp

from ffw_pendulum import generation_rollout


def main(net_config, train_config, log_config):
    """ Wrapper for CMA-ES search run. """
    # Start by setting the random seeds for reproducibility
    rng = set_random_seeds(train_config.seed_id, return_key=True)

    # Define logger, CMA-ES strategy
    train_log = DeepLogger(**log_config)

    net_config.network_size[1] = int(train_config.hidden_size)
    num_params = (net_config.network_size[0] * net_config.network_size[1]
                  + net_config.network_size[1]
                  + net_config.network_size[1]*net_config.network_size[2]
                  + net_config.network_size[2])
    mean_init = jnp.zeros(num_params)
    elite_size = int(train_config.pop_size * train_config.elite_percentage)
    es_params, es_memory = init_strategy(mean_init,
                                         train_config.sigma_init,
                                         train_config.pop_size,
                                         elite_size)
    evo_logger = init_logger(train_config.top_k, num_params)

    # Modify helper for gridsearch
    es_params = modify_cma_es_params(es_params, train_config)

    # Train the network using the training loop
    evo_logger = search_net(rng, elite_size,
                            train_config.num_generations,
                            train_config.num_evals_per_gen,
                            train_config.num_env_steps,
                            net_config.network_size,
                            dict(train_config.env_params),
                            es_params, es_memory, train_config.top_k,
                            evo_logger, train_log)

    # Save top k model array into numpy file
    np.save(os.path.join(train_log.experiment_dir, "networks",
                         "top_k_models_" + str(train_config.seed_id)
                         + ".npy"), evo_logger["top_params"])
    return


def modify_cma_es_params(es_params, train_config):
    """ Modify base config based on train config for gridsearch. """
    if train_config.c_m is not None:
        es_params["c_m"] = train_config.c_m
    if train_config.c_1 is not None:
        es_params["c_1"] = train_config.c_1
    if train_config.c_mu is not None:
        es_params["c_mu"] = train_config.c_mu
    if train_config.c_c is not None:
        es_params["c_c"] = train_config.c_c
    if train_config.c_sigma is not None:
        es_params["c_sigma"] = train_config.c_sigma
    return es_params


def search_net(rng, elite_size, num_generations, num_evals_per_gen,
               num_env_steps, network_size, env_params, es_params,
               es_memory, top_k, evo_logger, train_log):
    """ Run the training loop over a set of epochs. """
    # Print out the ES params before starting to train
    for key, value in es_params.items():
        if key != "weights":
              print(key, str(value))

    # Loop over different generations and search!
    for g in range(num_generations):
        rng, rng_input = jax.random.split(rng)
        x, es_memory = ask(rng_input, es_params, es_memory)
        generation_params = flat_to_mlp(x, sizes=network_size)

        # Evaluate the fitness of the generation members
        rng, rng_input = jax.random.split(rng)
        rollout_keys = jax.random.split(rng_input, num_evals_per_gen)
        population_returns = generation_rollout(rollout_keys,
                                                generation_params,
                                                env_params, num_env_steps)
        values = - population_returns.mean(axis=1)
        # Update the CMA-ES strategy
        es_memory = tell(x, values, es_params, es_memory)

        # Log current performance/pop stats and check termination!
        if (g + 1) % 5 == 0:
            evo_logger = update_logger(evo_logger, x, values,
                                       es_memory, top_k)
            time_tick = [g+1]
            stats_tick = [float(evo_logger["log_top_1"][-1]),
                          float(evo_logger["log_top_mean"][-1]),
                          float(evo_logger["log_top_std"][-1]),
                          float(evo_logger["log_gen_mean"][-1]),
                          float(evo_logger["log_gen_std"][-1]),
                          float(evo_logger["log_sigma"][-1])]
            train_log.update_log(time_tick, stats_tick)
            train_log.save_log()
            if check_termination(values, es_params, es_memory):
                break
    return evo_logger


if __name__ == "__main__":
    train_config, net_config, log_config = get_configs_ready(
        default_config_fname="configs/train/cma_config.json")
    main(net_config, train_config, log_config)
