import jax.numpy as jnp


def init_logger(top_k, num_params):
    evo_logger = {"top_values": jnp.zeros(top_k) + 1e10,
                  "top_params": jnp.zeros((top_k, num_params)),
                  "log_top_1": [],
                  "log_top_mean": [],
                  "log_top_std": [],
                  "log_gen_1": [],
                  "log_gen_mean": [],
                  "log_gen_std": [],
                  "log_sigma": [],
                  "log_gen": []}
    return evo_logger


def update_logger(evo_logger, x, fitness, memory, top_k, verbose=False):
    """ Helper function to keep track of top solutions. """
    # Check if there are solutions better than current archive
    vals = jnp.hstack([evo_logger["top_values"], fitness])
    params = jnp.vstack([evo_logger["top_params"], x])
    concat_top = jnp.hstack([jnp.expand_dims(vals, 1), params])
    sorted_top = concat_top[concat_top[:, 0].argsort()]

    # Importantly: Params are stored as flat vectors
    evo_logger["top_values"] = sorted_top[:top_k, 0]
    evo_logger["top_params"] = sorted_top[:top_k, 1:]
    evo_logger["log_top_1"].append(evo_logger["top_values"][0])
    evo_logger["log_top_mean"].append(jnp.mean(evo_logger["top_values"]))
    evo_logger["log_top_std"].append(jnp.std(evo_logger["top_values"]))
    evo_logger["log_gen_1"].append(jnp.min(fitness))
    evo_logger["log_gen_mean"].append(jnp.mean(fitness))
    evo_logger["log_gen_std"].append(jnp.std(fitness))
    evo_logger["log_sigma"].append(memory["sigma"])
    evo_logger["log_gen"].append(memory["generation"])
    if verbose:
        print(evo_logger["log_gen"][-1], evo_logger["top_values"])
    return evo_logger
