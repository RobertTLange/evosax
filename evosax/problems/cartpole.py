import jax
import jax.numpy as jnp
from jax import lax

from typing import Tuple
import chex
from functools import partial


class CartPole:
    """
    JAX Compatible version of CartPole-v1 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self):
        self.env_name = "CartPole-v1"
        self.action_shape = 2
        self.observation_shape = (4,)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: chex.ArrayTree,
        action: int,
        params: chex.ArrayTree,
    ) -> Tuple[chex.Array, chex.ArrayTree, float, bool]:
        """Performs step transitions in the environment."""
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key, params)
        return obs, state

    def discount(self, state: chex.ArrayTree, params: chex.ArrayTree) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def default_params(self) -> chex.ArrayTree:
        # Default environment parameters for CartPole-v1
        return {
            "gravity": 9.8,
            "masscart": 1.0,
            "masspole": 0.1,
            "total_mass": 1.0 + 0.1,  # (masscart + masspole)
            "length": 0.5,
            "polemass_length": 0.05,  # (masspole * length)
            "force_mag": 10.0,
            "tau": 0.02,
            "theta_threshold_radians": 12 * 2 * jnp.pi / 360,
            "x_threshold": 2.4,
            "max_steps_in_episode": 200,
        }

    def step_env(
        self,
        key: chex.PRNGKey,
        state: chex.ArrayTree,
        action: int,
        params: chex.ArrayTree,
    ) -> Tuple[chex.Array, chex.ArrayTree, float, bool, chex.ArrayTree]:
        """Performs step transitions in the environment."""
        force = params["force_mag"] * action - params["force_mag"] * (
            1 - action
        )
        costheta = jnp.cos(state["theta"])
        sintheta = jnp.sin(state["theta"])

        temp = (
            force
            + params["polemass_length"] * state["theta_dot"] ** 2 * sintheta
        ) / params["total_mass"]
        thetaacc = (params["gravity"] * sintheta - costheta * temp) / (
            params["length"]
            * (
                4.0 / 3.0
                - params["masspole"] * costheta ** 2 / params["total_mass"]
            )
        )
        xacc = (
            temp
            - params["polemass_length"]
            * thetaacc
            * costheta
            / params["total_mass"]
        )

        # Only default Euler integration option available here!
        x = state["x"] + params["tau"] * state["x_dot"]
        x_dot = state["x_dot"] + params["tau"] * xacc
        theta = state["theta"] + params["tau"] * state["theta_dot"]
        theta_dot = state["theta_dot"] + params["tau"] * thetaacc

        # Important: Reward is based on termination is previous step transition
        reward = 1.0 - state["terminal"]

        # Update state dict and evaluate termination conditions
        state = {
            "x": x,
            "x_dot": x_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "time": state["time"] + 1,
        }
        done = self.is_terminal(state, params)
        state["terminal"] = done

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """Performs resetting of environment."""
        init_state = jax.random.uniform(
            key, minval=-0.05, maxval=0.05, shape=(4,)
        )
        state = {
            "x": init_state[0],
            "x_dot": init_state[1],
            "theta": init_state[2],
            "theta_dot": init_state[3],
            "time": 0,
            "terminal": False,
        }
        return self.get_obs(state), state

    def get_obs(self, state: chex.ArrayTree) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array(
            [state["x"], state["x_dot"], state["theta"], state["theta_dot"]]
        )

    def is_terminal(
        self, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state["x"] < -params["x_threshold"],
            state["x"] > params["x_threshold"],
        )
        done2 = jnp.logical_or(
            state["theta"] < -params["theta_threshold_radians"],
            state["theta"] > params["theta_threshold_radians"],
        )

        # Check number of steps in episode termination condition
        done_steps = state["time"] > params["max_steps_in_episode"]
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done
