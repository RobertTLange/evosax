import jax
import jax.numpy as jnp
from jax import jit, vmap
from evosax.problems.pendulum import reset, step


def init_policy_mlp(rng_input, population_size, sizes, scale=1e-2):
    """ Initialize the weights of all layers of a relu + linear layer """
    # Initialize a single layer with Gaussian weights - helper function
    def initialize_layer(population_size, m, n, key, scale):
        w_key, b_key = jax.random.split(key)
        return (scale * jax.random.normal(w_key, (population_size, n, m)),
                scale * jax.random.normal(b_key, (population_size, n,)))

    keys = jax.random.split(rng_input, len(sizes)+1)
    W1, b1 = initialize_layer(population_size, sizes[0], sizes[1],
                              keys[0], scale)
    W2, b2 = initialize_layer(population_size, sizes[1], sizes[2],
                              keys[1], scale)
    if population_size == 1:
        params = {"W1": W1.squeeze(), "b1": b1.squeeze(), "W2": W2[0], "b2": b2[0]}
    else:
        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params


def ffw_policy(params, obs):
    """ Compute forward pass and return action from deterministic policy """
    def relu_layer(W, b, x):
        """ Simple ReLu layer for single sample """
        return jnp.maximum(0, (jnp.dot(W, x) + b))
    # Simple single hidden layer MLP: Obs -> Hidden -> Action
    activations = relu_layer(params["W1"], params["b1"], obs)
    mean_policy = jnp.dot(params["W2"], activations) + params["b2"]
    return mean_policy


def pendulum_rollout(rng_input, policy_params, env_params, num_steps):
    """ Rollout a pendulum episode with lax.scan. """
    obs, state = reset(rng_input)
    _, scan_out = jax.lax.scan(policy_pendulum_step,
                               [obs, state, policy_params, env_params],
                               [jnp.zeros(num_steps)])
    return jnp.sum(jnp.array(scan_out))


def policy_pendulum_step(state_input, tmp):
    """ lax.scan compatible step transition in jax env. """
    obs, state, policy_params, env_params = state_input
    action = ffw_policy(policy_params, obs)
    next_o, next_s, reward, done, _ = step(env_params, state, action)
    carry, y = [next_o.squeeze(), next_s.squeeze(),
                policy_params, env_params], [reward]
    return carry, y


def flat_to_network(flat_params, sizes):
    """ Reshape flat parameter vector to feedforward network param dict. """
    pop_size = flat_params.shape[0]
    W1_stop = sizes[0]*sizes[1]
    b1_stop = W1_stop + sizes[1]
    W2_stop = b1_stop + (sizes[1]*sizes[2])
    b2_stop = W2_stop + sizes[2]
    # Reshape params into weight/bias shapes
    params = {"W1": flat_params[:, :W1_stop].reshape(pop_size,
                                                     sizes[1], sizes[0]),
              "b1": flat_params[:, W1_stop:b1_stop],
              "W2": flat_params[:, b1_stop:W2_stop].reshape(pop_size,
                                                            sizes[2], sizes[1]),
              "b2": flat_params[:, W2_stop:b2_stop]}
    return params


batch_rollout = jit(vmap(pendulum_rollout, in_axes=(0, None, None, None),
                         out_axes=0), static_argnums=(3))
v_dict = {"W1": 0, "b1": 0, "W2": 0, "b2": 0}
generation_rollout = jit(vmap(batch_rollout,
                              in_axes=(None, v_dict, None, None),
                              out_axes=0), static_argnums=(2, 3))
