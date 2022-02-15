import jax
import jax.numpy as jnp
from brax import envs
import chex
from typing import Tuple


class BraxFitness(object):
    def __init__(self, problem_name: str = "ant", num_env_steps: int = 1000):
        self.env_name = problem_name
        self.num_env_steps = num_env_steps
        assert self.env_name in [
            "ant",
            "halfcheetah",
            "hopper",
            "humanoid",
            "reacher",
            "walker2d",
            "fetch",
            "grasp",
            "ur5e",
        ]

        # Define the RL environment & network forward fucntion
        self.env = envs.create(env_name=self.env_name)
        self.action_shape = self.env.action_size

    def set_apply_fn(self, network, carry_init=None):
        """Set the network forward function."""
        self.network = network
        # Set rollout function based on model architecture
        if carry_init is not None:
            self.single_rollout = self.rollout_rnn
            self.carry_init = carry_init
        else:
            self.rollout = self.rollout_ffw

    def rollout_ffw(
        self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree
    ) -> chex.Array:
        """Rollout a jitted brax episode with lax.scan for a feedforward policy."""
        # Reset the environment
        rng, rng_reset = jax.random.split(rng_input)
        state = self.env.reset(rng_reset)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            state, policy_params, rng = state_input
            rng, rng_net = jax.random.split(rng)
            action = self.network.apply(
                {"params": policy_params}, state.obs, rng=rng_net
            )
            next_s = self.env.step(state, action)
            carry = [next_s, policy_params, rng]
            return carry, [state.reward, state.done]

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step,
            [state, policy_params, rng],
            [jnp.zeros((self.num_env_steps, 2))],
        )
        # Return masked sum of rewards accumulated by agent in episode
        rewards, dones = scan_out[0], scan_out[1]
        rewards = rewards.reshape(self.num_env_steps, 1)
        ep_mask = (jnp.cumsum(dones) < 1).reshape(self.num_env_steps, 1)
        return jnp.sum(rewards * ep_mask)

    def rollout_rnn(
        self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree
    ) -> chex.Array:
        """Rollout a jitted episode with lax.scan for a recurrent policy."""
        # Reset the environment
        rng, rng_reset = jax.random.split(rng_input)
        state = self.env.reset(rng_reset)
        hidden = self.carry_init()

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            state, policy_params, rng, hidden = state_input
            rng, rng_net = jax.random.split(rng)
            hidden, action = self.network.apply(
                {"params": policy_params}, state.obs, hidden, rng_net
            )
            next_s = self.env.step(state, action)
            carry = [next_s, policy_params, rng, hidden]
            return carry, [state.reward, state.done]

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step,
            [state, policy_params, rng, hidden],
            [jnp.zeros((self.num_env_steps, 2))],
        )
        # Return masked sum of rewards accumulated by agent in episode
        rewards, dones = scan_out[0], scan_out[1]
        rewards = rewards.reshape(self.num_env_steps, 1)
        ep_mask = (jnp.cumsum(dones) < 1).reshape(self.num_env_steps, 1)
        return jnp.sum(rewards * ep_mask)

    @property
    def input_shape(self) -> Tuple[int]:
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        state = self.env.reset(rng)
        return state.obs.shape
