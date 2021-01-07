import jax
import jax.numpy as jnp
from jax import jit
from flax.core import FrozenDict


params = FrozenDict({"sample_probs": jnp.array([0.1, 0.9]),
                     "num_arms": 2,
                     "max_steps": 100})


def step_bandit(rng_input, params, state, action):
    """ Sample bernoulli reward, increase counter, construct input. """
    time = state[2] + 1
    done = (time >= params["max_steps"])
    reward = jax.random.bernoulli(rng_input, state[action])
    obs = get_obs_bandit(reward, action, time, params)
    state = jax.ops.index_update(state, 2, time)
    return obs, state, reward, done, {}


def reset_bandit(rng_input, params):
    """ Reset the Bernoulli bandit. Resample arm identities. """
    # Sample reward function + construct state as concat with timestamp
    p1 = jax.random.choice(rng_input, params["sample_probs"],
                           shape=(1,)).squeeze()
    # State representation: Mean reward a1, Mean reward a2, t
    state = jnp.stack([p1, 1-p1, 0])
    return get_obs_bandit(0, 0, 0, params), state


def get_obs_bandit(reward, action, time, params):
    """ Concatenate reward, one-hot action and time stamp. """
    action_one_hot = jax.nn.one_hot(action, params["num_arms"]).squeeze()
    return jnp.hstack([reward, action_one_hot, time])


reset = jit(reset_bandit, static_argnums=(1,))
step = jit(step_bandit, static_argnums=(1,))
