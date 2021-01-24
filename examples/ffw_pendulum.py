import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from evosax.problems.pendulum import reset, step


def init_policy_mlp(rng_input, sizes, scale=1e-2):
    """ Initialize the weights of all layers of a relu + linear layer """
    # Initialize a single layer with Gaussian weights - helper function
    def initialize_layer(m, n, key, scale):
        w_key, b_key = jax.random.split(key)
        return (scale * jax.random.normal(w_key, (n, m)),
                scale * jax.random.normal(b_key, (n,)))

    keys = jax.random.split(rng_input, len(sizes)+1)
    W1, b1 = initialize_layer(sizes[0], sizes[1], keys[0], scale)
    W2, b2 = initialize_layer(sizes[1], sizes[2], keys[1], scale)

    params = {"layer_in": {"W1": W1, "b1": b1},
              "layer_out": {"W2": W2, "b2": b2}}
    return params


def ffw_policy(params, obs):
    """ Compute forward pass and return action from deterministic policy """
    def relu_layer(W, b, x):
        """ Simple ReLu layer for single sample """
        return jnp.maximum(0, (jnp.dot(W, x) + b))
    # Simple single hidden layer MLP: Obs -> Hidden -> Action
    activations = relu_layer(params["layer_in"]["W1"],
                             params["layer_in"]["b1"], obs)
    mean_policy = jnp.dot(params["layer_out"]["W2"],
                          activations) + params["layer_out"]["b2"]
    return mean_policy


def pendulum_rollout(rng_input, policy_params, env_params):
    """ Rollout a pendulum episode with lax.scan. """
    obs, state = reset(rng_input)
    _, scan_out = jax.lax.scan(policy_pendulum_step,
                               [obs, state, policy_params, env_params],
                               [jnp.zeros(env_params["max_steps_in_episode"])])
    return jnp.sum(jnp.array(scan_out))


def policy_pendulum_step(state_input, tmp):
    """ lax.scan compatible step transition in jax env. """
    obs, state, policy_params, env_params = state_input
    action = ffw_policy(policy_params, obs)
    next_o, next_s, reward, done, _ = step(env_params, state, action)
    carry, y = [next_o.squeeze(), next_s.squeeze(),
                policy_params, env_params], [reward]
    return carry, y


batch_rollout_no_jit = vmap(pendulum_rollout, in_axes=(0, None, None, None),
                            out_axes=0)
batch_rollout = jit(batch_rollout_no_jit, static_argnums=(3))

v_dict = {"W1": 0, "b1": 0, "W2": 0, "b2": 0}

generation_rollout_no_jit = vmap(batch_rollout_no_jit,
                                 in_axes=(None, v_dict, None, None),
                                 out_axes=0)
generation_rollout = jit(vmap(batch_rollout,
                              in_axes=(None, v_dict, None, None),
                              out_axes=0), static_argnums=(2, 3))

generation_rollout_pmap = jit(pmap(batch_rollout,
                              in_axes=(None, v_dict, None, None)),
                              static_argnums=(2, 3))
