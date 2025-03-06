"""Brax Problem for Reinforcement Learning Optimization.

This module implements a problem class for reinforcement learning optimization using
the Brax library, which provides JAX-compatible physics-based RL environments.

The BraxProblem class handles:
- Environment setup and configuration
- Policy network evaluation through environment rollouts
- Tracking of environment interactions
- Support for both feedforward and recurrent neural network policies

[1] https://github.com/google/brax
"""

from functools import partial

import jax
import jax.numpy as jnp
from evosax.types import Fitness, PyTree, Solution
from flax import linen as nn

from ..problem import Problem


class BraxProblem(Problem):
    """Brax Problem for Reinforcement Learning Optimization."""

    def __init__(
        self,
        env_name: str,
        policy: nn.Module,
        episode_length: int | None = None,
        num_rollouts: int = 1,
        env_kwargs: dict = {"backend": "spring"},
    ):
        """Initialize the Brax problem."""
        try:
            import brax.envs
        except ImportError:
            raise ImportError("You need to install `brax` to use this problem class.")

        self.env_name = env_name
        self.policy = policy
        self.num_rollouts = num_rollouts

        # Instantiate environment
        self.env = brax.envs.create(
            env_name=self.env_name,
            episode_length=episode_length,
            auto_reset=True,
            **env_kwargs,
        )

        # Set episode length
        if episode_length is None:
            self.episode_length = self.env.episode_length
        else:
            self.episode_length = episode_length

        # Test policy and env compatibility
        key = jax.random.key(0)
        state = self.env.reset(key)
        policy_params = self.policy.init(key, state.obs, key)
        action = self.policy.apply(policy_params, state.obs, key)
        next_state = self.env.step(state, action)

        # Pegasus trick
        self._rollouts = jax.vmap(self._rollout, in_axes=(0, None))
        self._eval = jax.vmap(self._rollouts, in_axes=(None, 0))

    @property
    def observation_shape(self):
        """Observation shape of the environment."""
        return (self.env.observation_size,)

    @property
    def action_shape(self):
        """Action shape of the environment."""
        return (self.env.action_size,)

    @partial(jax.jit, static_argnames=("self",))
    def eval(self, key: jax.Array, solutions: Solution) -> tuple[Fitness, PyTree]:
        """Evaluate a population of policies."""
        keys = jax.random.split(key, self.num_rollouts)
        fitness, states = self._eval(keys, solutions)
        return jnp.mean(fitness, axis=-1), states

    def _rollout(self, key: jax.Array, policy_params: PyTree):
        """Perform a single rollout in the environment."""
        key_reset, key_scan = jax.random.split(key)

        # Reset environment
        state = self.env.reset(key_reset)

        def _step(carry, key):
            state, cum_reward, valid = carry

            # Sample action from policy
            action = self.policy.apply(policy_params, state.obs, key)

            # Step environment
            next_state = self.env.step(state, action)

            # Update cumulative reward and valid mask
            next_cum_reward = cum_reward + state.reward * valid
            next_valid = valid * (1 - state.done)
            carry = (
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
                state,
                jnp.array(0.0),
                jnp.array(1.0),
            ),
            xs=keys,
        )

        # Return the sum of rewards accumulated by agent in episode rollout and states
        return carry[1], states

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        key_init, key_reset, key_input = jax.random.split(key, 3)
        state = self.env.reset(key_reset)
        return self.policy.init(key_init, state.obs, key_input)
