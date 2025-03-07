"""Brax Problem for Reinforcement Learning Optimization.

This module implements a problem class for reinforcement learning optimization using
the Brax library, which provides JAX-compatible physics-based RL environments.

The BraxProblem class handles:
- Environment setup and configuration
- Policy network evaluation through environment rollouts
- Tracking of environment interactions
- Support for both feedforward and recurrent neural network policies

[1] https://github.com/google/brax
[2] https://github.com/google/brax/blob/main/brax/training/acme/running_statistics.py
"""

from functools import partial

import jax
import jax.numpy as jnp
from evosax.types import Fitness, Metrics, Population, PyTree, Solution
from flax import linen as nn
from flax import struct

from ..problem import Problem, State


@struct.dataclass
class State(State):
    obs_mean: PyTree
    obs_std: PyTree
    obs_var_sum: PyTree
    obs_counter: int
    std_min: float
    std_max: float


class BraxProblem(Problem):
    """Brax Problem for Reinforcement Learning Optimization."""

    def __init__(
        self,
        env_name: str,
        policy: nn.Module,
        episode_length: int | None = None,
        num_rollouts: int = 1,
        use_normalize_obs: bool = True,
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
        self.use_normalize_obs = use_normalize_obs

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
        env_state = self.env.reset(key)

        policy_params = self.policy.init(key, env_state.obs, key)

        action = self.policy.apply(policy_params, env_state.obs, key)
        self.env.step(env_state, action)

        # Pegasus trick
        self._rollouts = jax.vmap(self._rollout, in_axes=(0, None, None))
        self._eval = jax.vmap(self._rollouts, in_axes=(None, 0, None))

    @property
    def observation_shape(self):
        """Observation shape of the environment."""
        return (self.env.observation_size,)

    @property
    def action_shape(self):
        """Action shape of the environment."""
        return (self.env.action_size,)

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array) -> State:
        """Initialize state with empty normalization statistics."""
        # Create a dummy environment state to get the observation structure
        dummy_env_state = self.env.reset(key)

        return State(
            counter=0,
            obs_mean=jax.tree_map(lambda x: jnp.zeros_like(x), dummy_env_state.obs),
            obs_std=jax.tree_map(lambda x: jnp.ones_like(x), dummy_env_state.obs),
            obs_var_sum=jax.tree_map(lambda x: jnp.zeros_like(x), dummy_env_state.obs),
            obs_counter=0,
            std_min=1e-6,
            std_max=1e6,
        )

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self, key: jax.Array, solutions: Population, state: State
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a population of policies."""
        keys = jax.random.split(key, self.num_rollouts)
        fitness, env_states = self._eval(keys, solutions, state)

        # Update running statistics
        if self.use_normalize_obs:
            state = self.update_stats(env_states.obs, state)

        return (
            jnp.mean(fitness, axis=-1),
            state.replace(counter=state.counter + 1),
            {"env_states": env_states},
        )

    def _rollout(
        self, key: jax.Array, policy_params: PyTree, state: State
    ) -> tuple[jax.Array, PyTree]:
        """Perform a single rollout in the environment."""
        key_reset, key_scan = jax.random.split(key)

        # Reset environment
        env_state = self.env.reset(key_reset)

        def _step(carry, key):
            env_state, cum_reward, valid = carry

            # Normalize observations
            obs = self.normalize_obs(env_state.obs, state)

            # Sample action from policy
            action = self.policy.apply(policy_params, obs, key)

            # Step environment
            env_state = self.env.step(env_state, action)

            # Update cumulative reward and valid mask
            cum_reward = cum_reward + env_state.reward * valid
            valid = valid * (1 - env_state.done)
            carry = (
                env_state,
                cum_reward,
                valid,
            )
            return carry, env_state

        # Rollout
        keys = jax.random.split(key_scan, self.episode_length)
        carry, env_states = jax.lax.scan(
            _step,
            (
                env_state,
                jnp.array(0.0),
                jnp.array(1.0),
            ),
            xs=keys,
        )

        # Return the sum of rewards accumulated by agent in episode rollout and states
        return carry[1], env_states

    def normalize_obs(self, obs: PyTree, state: State) -> PyTree:
        """Normalize observations using running statistics."""
        return jax.tree_map(
            lambda obs, mean, std: (obs - mean) / std,
            obs,
            state.obs_mean,
            state.obs_std,
        )

    def update_stats(self, obs: PyTree, state: State) -> State:
        """Update running statistics for observations using Welford's online algorithm.

        This method implements a numerically stable algorithm for computing
        running mean and variance statistics across episodes [2].

        Args:
            obs: PyTree containing observations with shape
                (population_size, num_rollouts, episode_length, ...)
            state: Current state containing running statistics

        Returns:
            Updated state with new observation statistics

        """
        # Batch dimensions are (population_size, num_rollouts, episode_length)
        batch_size = obs.shape[0] * obs.shape[1] * obs.shape[2]
        new_obs_counter = state.obs_counter + batch_size

        # Function to update statistics for each leaf in the PyTree
        def _update_leaf_stats(leaf_obs, leaf_mean, leaf_var_sum):
            # Compute the new mean
            diff_to_old_mean = leaf_obs - leaf_mean
            new_obs_mean = (
                leaf_mean + jnp.sum(diff_to_old_mean, axis=(0, 1, 2)) / new_obs_counter
            )

            # Compute new variance
            diff_to_new_mean = leaf_obs - new_obs_mean
            new_obs_var_sum = leaf_var_sum + jnp.sum(
                diff_to_old_mean * diff_to_new_mean, axis=(0, 1, 2)
            )

            return new_obs_mean, new_obs_var_sum

        # Apply the update function to each leaf in the observation PyTree
        obs_mean, obs_var_sum = jax.tree.map(
            lambda obs, mean, var: _update_leaf_stats(obs, mean, var),
            obs,
            state.obs_mean,
            state.obs_var_sum,
        )

        obs_var_sum = jnp.maximum(obs_var_sum, 0)
        obs_std = jnp.sqrt(obs_var_sum / new_obs_counter)
        obs_std = jnp.clip(obs_std, state.std_min, state.std_max)

        # Return updated state with new statistics
        return state.replace(
            obs_mean=obs_mean,
            obs_std=obs_std,
            obs_var_sum=obs_var_sum,
            obs_counter=new_obs_counter,
        )

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        key_init, key_reset, key_input = jax.random.split(key, 3)
        env_state = self.env.reset(key_reset)
        return self.policy.init(key_init, env_state.obs, key_input)
