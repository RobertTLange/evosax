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
from evosax.types import Fitness, Metrics, PyTree, Solution
from flax import linen as nn, struct

from ..problem import Problem, State


@struct.dataclass
class State(State):
    obs_mean: PyTree
    obs_std: PyTree
    obs_var_sum: PyTree
    obs_counter: int
    std_min: float
    std_max: float


class GymnaxProblem(Problem):
    """Gymnax Problem for Reinforcement Learning Optimization."""

    def __init__(
        self,
        env_name: str,
        policy: nn.Module,
        episode_length: int | None = None,
        num_rollouts: int = 1,
        use_normalize_obs: bool = True,
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
        self.use_normalize_obs = use_normalize_obs

        # Instantiate environment and replace default parameters
        self.env, self.env_params = gymnax.make(self.env_name, **env_kwargs)
        self.env_params.replace(**env_params)

        # Test policy and env compatibility
        key = jax.random.key(0)
        obs, state = self.env.reset(key, self.env_params)

        policy_params = self.policy.init(key, obs, key)

        action = self.policy.apply(policy_params, obs, key)
        self.env.step(key, state, action, self.env_params)

        # Set number of environment steps
        if episode_length is None:
            self.episode_length = self.env_params.max_steps_in_episode
        else:
            self.episode_length = episode_length

        # Pegasus trick
        self._rollouts = jax.vmap(self._rollout, in_axes=(0, None, None))
        self._eval = jax.vmap(self._rollouts, in_axes=(None, 0, None))

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
    def init(self, key: jax.Array) -> State:
        """Initialize state with empty normalization statistics."""
        # Create a dummy observation to get the observation structure
        dummy_obs, _ = self.env.reset(key, self.env_params)

        return State(
            counter=0,
            obs_mean=jax.tree.map(lambda x: jnp.zeros_like(x), dummy_obs),
            obs_std=jax.tree.map(lambda x: jnp.ones_like(x), dummy_obs),
            obs_var_sum=jax.tree.map(lambda x: jnp.zeros_like(x), dummy_obs),
            obs_counter=0,
            std_min=1e-6,
            std_max=1e6,
        )

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self, key: jax.Array, solutions: Solution, state: State
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a population of policies."""
        keys = jax.random.split(key, self.num_rollouts)
        fitness, env_states = self._eval(keys, solutions, state)

        # Update running statistics
        if self.use_normalize_obs:
            state = self.update_stats(env_states[0], state)

        return (
            jnp.mean(fitness, axis=-1),
            state.replace(counter=state.counter + 1),
            {"env_states": env_states},
        )

    def _rollout(self, key: jax.Array, policy_params: PyTree, state: State):
        key_reset, key_scan = jax.random.split(key)

        # Reset environment
        obs, env_state = self.env.reset(key_reset, self.env_params)

        def _step(carry, key):
            obs, env_state, cum_reward, valid = carry

            key_action, key_step = jax.random.split(key)

            # Normalize observations
            obs = self.normalize_obs(obs, state) if self.use_normalize_obs else obs

            # Sample action from policy
            action = self.policy.apply(policy_params, obs, key_action)

            # Step environment
            obs, env_state, reward, done, _ = self.env.step(
                key_step, env_state, action, self.env_params
            )

            # Update cumulative reward and valid mask
            cum_reward = cum_reward + reward * valid
            valid = valid * (1 - done)
            carry = (
                obs,
                env_state,
                cum_reward,
                valid,
            )
            return carry, (obs, env_state)

        # Rollout
        keys = jax.random.split(key_scan, self.episode_length)
        carry, env_states = jax.lax.scan(
            _step,
            (
                obs,
                env_state,
                jnp.array(0.0),
                jnp.array(1.0),
            ),
            xs=keys,
        )

        # Return the sum of rewards accumulated by agent in episode rollout and states
        return carry[2], env_states

    def normalize_obs(self, obs: PyTree, state: State) -> PyTree:
        """Normalize observations using running statistics."""
        return jax.tree.map(
            lambda obs, mean, std: (obs - mean) / std,
            obs,
            state.obs_mean,
            state.obs_std,
        )

    def update_stats(self, obs: PyTree, state: State) -> State:
        """Update running statistics for observations using Welford's online algorithm.

        This method implements a numerically stable algorithm for computing
        running mean and variance statistics across episodes.

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
        key_init, key_sample, key_input = jax.random.split(key, 3)
        obs = self.observation_space.sample(key_sample)
        return self.policy.init(key_init, obs, key_input)
