import jax
import jax.numpy as jnp
from jax import jit
from flax.core import FrozenDict

# Default environment parameters for Pendulum-v0
env_params = FrozenDict({"max_speed": 8,
                         "max_torque": 2.,
                         "dt": 0.05,
                         "g": 10.0,
                         "m": 1.,
                         "l": 1.,
                         "max_steps_in_episode": 200})


def step_pendulum(params, state, u):
    """ Integrate pendulum ODE and return transition. """
    th, thdot = state[0], state[1]
    u = jnp.clip(u, -params["max_torque"], params["max_torque"])
    costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

    newthdot = thdot + (-3 * params["g"] /
                        (2 * params["l"]) * jnp.sin(th + jnp.pi) + 3. /
                        (params["m"] * params["l"] ** 2) * u) * params["dt"]
    newth = th + newthdot * params["dt"]
    newthdot = jnp.clip(newthdot, -params["max_speed"], params["max_speed"])

    state = jnp.array([newth, newthdot])
    return get_obs_pendulum(state), state, -costs[0].squeeze(), False, {}


def reset_pendulum(rng):
    """ Reset environment state by sampling theta, thetadot. """
    high = jnp.array([jnp.pi, 1])
    state = jax.random.uniform(rng, shape=(2,),
                               minval=-high, maxval=high)
    return get_obs_pendulum(state), state


reset = jit(reset_pendulum)
step = jit(step_pendulum)


def get_obs_pendulum(state):
    """ Return angle in polar coordinates and change. """
    th, thdot = state[0], state[1]
    return jnp.array([jnp.cos(th), jnp.sin(th), thdot]).squeeze()


def angle_normalize(x):
    """ Normalize the angle - radians. """
    return (((x+jnp.pi) % (2*jnp.pi)) - jnp.pi)
