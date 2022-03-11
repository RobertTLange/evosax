import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional
import chex
from .cartpole import CartPole


class GymFitness(object):
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        num_env_steps: int = 200,
        num_rollouts: int = 16,
        test: bool = False,
        n_devices: Optional[int] = None,
    ):
        self.env_name = env_name
        self.num_env_steps = num_env_steps
        self.num_rollouts = num_rollouts
        self.steps_per_member = num_env_steps * num_rollouts
        self.test = test

        # Define the RL environment & network forward fucntion
        self.env = CartPole()
        self.env_params = self.env.default_params
        self.action_shape = self.env.action_shape
        self.input_shape = self.env.observation_shape
        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices

    def set_apply_fn(self, map_dict, network_apply, carry_init=None):
        """Set the network forward function."""
        self.network = network_apply
        # Set rollout function based on model architecture
        if carry_init is not None:
            self.single_rollout = self.rollout_rnn
            self.carry_init = carry_init
        else:
            self.single_rollout = self.rollout_ffw
        self.rollout_repeats = jax.vmap(self.single_rollout, in_axes=(0, None))
        self.rollout_pop = jax.vmap(
            self.rollout_repeats, in_axes=(None, map_dict)
        )
        # pmap over popmembers if > 1 device is available - otherwise pmap
        if self.n_devices > 1:
            self.rollout_map = self.rollout_pmap
            print(
                "More than one device detected. Please make sure that the ES"
                " population size divides evenly across the number of devices"
                " to pmap/parallelize over."
            )
        else:
            self.rollout_map = self.rollout_pop

    def rollout_pmap(
        self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree
    ):
        """Parallelize rollout across devices. Split keys/reshape correctly."""
        keys_pmap = jnp.tile(rng_input, (self.n_devices, 1, 1))
        rew_dev = jax.pmap(self.rollout_pop)(keys_pmap, policy_params)
        rew_re = rew_dev.reshape(-1, self.num_rollouts)
        return rew_re

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree):
        """Placeholder fn call for rolling out a population for multi-evals."""
        rng_pop = jax.random.split(rng_input, self.num_rollouts)
        return self.rollout_map(rng_pop, policy_params)

    def rollout_ffw(
        self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree
    ):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action = self.network({"params": policy_params}, obs, rng=rng_net)
            # action = jnp.nan_to_num(action, nan=0.0)
            next_o, next_s, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            carry, y = [next_o.squeeze(), next_s, policy_params, rng], [
                reward,
                done,
            ]
            return carry, y

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step,
            [obs, state, policy_params, rng_episode],
            [jnp.zeros((self.num_env_steps, 2))],
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        rewards, dones = scan_out[0], scan_out[1]
        rewards = rewards.reshape(self.num_env_steps, 1)
        ep_mask = (jnp.cumsum(dones) < 1).reshape(self.num_env_steps, 1)
        return jnp.sum(rewards * ep_mask)

    def rollout_rnn(
        self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree
    ):
        """Rollout a jitted episode with lax.scan."""
        # Reset the environment
        rng, rng_reset = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)
        hidden = self.carry_init()

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng, hidden = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            hidden, action = self.network(
                {"params": policy_params}, obs, hidden, rng_net
            )
            next_o, next_s, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            carry, y = [next_o.squeeze(), next_s, policy_params, rng, hidden], [
                reward,
                done,
            ]
            return carry, y

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step,
            [obs, state, policy_params, rng, hidden],
            [jnp.zeros((self.num_env_steps, 2))],
        )
        # Return masked sum of rewards accumulated by agent in episode
        rewards, dones = scan_out[0], scan_out[1]
        rewards = rewards.reshape(self.num_env_steps, 1)
        ep_mask = (jnp.cumsum(dones) < 1).reshape(self.num_env_steps, 1)
        return jnp.sum(rewards * ep_mask)
