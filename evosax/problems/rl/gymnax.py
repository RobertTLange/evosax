"""Gymnax Problem for Reinforcement Learning Optimization.

This module implements a problem class for reinforcement learning optimization using
the Gymnax library, which provides JAX-compatible RL environments.

The GymnaxProblem class handles:
- Environment setup and configuration
- Policy network evaluation through environment rollouts
- Tracking of environment interactions
- Support for both feedforward and recurrent neural network policies

[1] https://github.com/RobertTLange/gymnax
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from ...types import Fitness, PyTree, Solution
from ..problem import Problem


class GymnaxProblem(Problem):
    """Gymnax Problem for Reinforcement Learning Optimization."""

    def __init__(
        self,
        env_name: str,
        policy: nn.Module,
        episode_length: int | None = None,
        num_rollouts: int = 1,
        env_kwargs: dict = {},
        env_params: dict = {},
    ):
        """Initialize the Gymnax problem."""
        try:
            import gymnax
        except ImportError:
            raise ImportError("You need to install `gymnax` to use this problem class.")

        self.env_name = env_name
        self.policy = policy
        self.num_rollouts = num_rollouts

        # Instantiate environment and replace default parameters
        self.env, self.env_params = gymnax.make(self.env_name, **env_kwargs)
        self.env_params.replace(**env_params)

        # Test policy and env compatibility
        key = jax.random.key(0)
        obs, state = self.env.reset(key, self.env_params)
        policy_params = self.policy.init(key, obs, key)
        action = self.policy.apply(policy_params, obs, key)
        next_obs, next_state, reward, done, _ = self.env.step(
            key, state, action, self.env_params
        )

        # Set number of environment steps
        if episode_length is None:
            self.episode_length = self.env_params.max_steps_in_episode
        else:
            self.episode_length = episode_length

        # Pegasus trick
        self._rollouts = jax.vmap(self._rollout, in_axes=(0, None))
        self._eval = jax.vmap(self._rollouts, in_axes=(None, 0))

    @property
    def observation_space(self):
        """Observation space of the environment."""
        return self.env.observation_space(self.env_params)

    @property
    def action_space(self):
        """Action space of the environment."""
        return self.env.action_space(self.env_params)

    @property
    def observation_shape(self):
        """Observation shape of the environment."""
        return self.observation_space.shape

    @property
    def action_shape(self):
        """Action shape of the environment."""
        return self.action_space.shape

    @property
    def num_actions(self):
        """Number of actions in the environment."""
        return self.env.num_actions

    @partial(jax.jit, static_argnames=("self",))
    def eval(self, key: jax.Array, solutions: Solution) -> tuple[Fitness, PyTree]:
        """Evaluate a population of policies."""
        keys = jax.random.split(key, self.num_rollouts)
        fitness, states = self._eval(keys, solutions)
        return jnp.mean(fitness, axis=-1), states

    def _rollout(self, key: jax.Array, policy_params: PyTree):
        key_reset, key_scan = jax.random.split(key)

        # Reset environment
        obs, state = self.env.reset(key_reset, self.env_params)

        def _step(carry, key):
            obs, state, cum_reward, valid = carry

            key_action, key_step = jax.random.split(key)

            # Sample action from policy
            action = self.policy.apply(policy_params, obs, key_action)

            # Step environment
            next_obs, next_state, reward, done, _ = self.env.step(
                key_step, state, action, self.env_params
            )

            # Update cumulative reward and valid mask
            next_cum_reward = cum_reward + reward * valid
            next_valid = valid * (1 - done)
            carry = (
                next_obs,
                next_state,
                next_cum_reward,
                next_valid,
            )
            return carry, next_state

        # Rollout
        keys = jax.random.split(key_scan, self.episode_length)
        carry, states = jax.lax.scan(
            _step,
            (
                obs,
                state,
                jnp.array(0.0),
                jnp.array(1.0),
            ),
            xs=keys,
        )

        # Return the sum of rewards accumulated by agent in episode rollout and states
        return carry[2], states

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        key_init, key_sample, key_input = jax.random.split(key, 3)
        obs = self.observation_space.sample(key_sample)
        return self.policy.init(key_init, obs, key_input)
