from mle_toolbox.utils import set_random_seeds, get_configs_ready, DeepLogger

import os
import jax
import jax.numpy as jnp
import numpy as np

import gymnax
from gymnax.rollouts import DeterministicRollouts

from evosax.strategies.cma_es import init_strategy, ask, tell, check_termination
from evosax.utils import (
    init_logger,
    update_logger,
    flat_to_network,
    get_total_params,
    get_network_shapes,
)
from ffw_pendulum import ffw_policy, init_policy_mlp


def main(net_config, train_config, log_config):
    """Wrapper for CMA-ES search run."""
    # Start by setting the random seeds for reproducibility
    rng = set_random_seeds(train_config.seed_id, return_key=True)

    # Define logger
    train_log = DeepLogger(**log_config)

    # Init placeholder net to get total no params and network layout
    net_config.network_size[1] = int(train_config.hidden_size)
    policy_params = init_policy_mlp(rng, net_config.network_size)
    total_no_params = get_total_params(policy_params)
    network_shapes = get_network_shapes(policy_params)
    mean_init = jnp.zeros(total_no_params)

    # Define/Init CMA-ES strategy
    elite_size = int(train_config.pop_size * train_config.elite_percentage)
    es_params, es_memory = init_strategy(
        mean_init, train_config.sigma_init, train_config.pop_size, elite_size
    )
    evo_logger = init_logger(train_config.top_k, total_no_params, network_shapes)

    # Modify helper for gridsearch
    es_params = modify_cma_es_params(es_params, train_config)

    # Train the network using the training loop
    evo_logger = search_net(
        rng,
        train_config.num_generations,
        train_config.num_evals_per_gen,
        es_params,
        es_memory,
        network_shapes,
        train_config.top_k,
        evo_logger,
        train_log,
    )

    # Save top k model array into numpy file
    np.save(
        os.path.join(
            train_log.experiment_dir,
            "networks",
            "top_k_models_" + str(train_config.seed_id) + ".npy",
        ),
        evo_logger["top_params"],
    )
    return


def modify_cma_es_params(es_params, train_config):
    """Modify base config based on train config for gridsearch."""
    if "c_m" in train_config.keys():
        es_params["c_m"] = train_config.c_m
    if "c_1" in train_config.keys():
        es_params["c_1"] = train_config.c_1
    if "c_mu" in train_config.keys():
        es_params["c_mu"] = train_config.c_mu
    if "c_c" in train_config.keys():
        es_params["c_c"] = train_config.c_c
    if "c_sigma" in train_config.keys():
        es_params["c_sigma"] = train_config.c_sigma
    return es_params


def search_net(
    rng,
    num_generations,
    num_evals_per_gen,
    es_params,
    es_memory,
    network_shapes,
    top_k,
    evo_logger,
    train_log,
):
    """Run the training loop over a set of epochs."""
    # Setup the gymnax rollout collector for pendulum
    _, reset, step, env_params = gymnax.make("Pendulum-v0")
    collector = DeterministicRollouts(ffw_policy, step, reset, env_params)
    collector.init_collector()

    # Wrap reshaping of param vector and rollout into single fct. to vmap
    def reshape_and_eval(rng, x, network_shapes):
        """Perform both parameter reshaping and evaluation in one go."""
        net_params = flat_to_network(x, network_shapes)
        traces, returns = collector.batch_rollout(rng, net_params)
        return -returns.mean(axis=0).sum()

    # vmap the merged reshape + rollout function over population members
    generation_rollout = jax.vmap(reshape_and_eval, in_axes=(None, 0, None))

    # Print out the ES params before starting to train
    for key, value in es_params.items():
        if key != "weights":
            print(key, str(value))

    # Loop over different generations and search!
    for g in range(num_generations):
        # Ask ES for next generation params
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        x, es_memory = ask(rng_ask, es_params, es_memory)

        # Reshape and rollout/eval batch
        rollout_keys = jax.random.split(rng_eval, num_evals_per_gen)
        values = generation_rollout(rollout_keys, x, network_shapes)

        # Update ES, log + check for termination
        es_memory = tell(x, values, es_params, es_memory)

        # Log current performance/pop stats and check termination!
        if (g + 1) % 5 == 0:
            evo_logger = update_logger(evo_logger, x, values, es_memory, top_k)
            time_tick = [g + 1]
            stats_tick = [
                float(evo_logger["log_top_1"][-1]),
                float(evo_logger["log_top_mean"][-1]),
                float(evo_logger["log_top_std"][-1]),
                float(evo_logger["log_gen_mean"][-1]),
                float(evo_logger["log_gen_std"][-1]),
                float(evo_logger["log_sigma"][-1]),
            ]
            train_log.update_log(time_tick, stats_tick)
            train_log.save_log()
            if check_termination(values, es_params, es_memory):
                break
    return evo_logger


if __name__ == "__main__":
    train_config, net_config, log_config = get_configs_ready(
        default_config_fname="configs/train/cma_config.json"
    )
    main(net_config, train_config, log_config)
