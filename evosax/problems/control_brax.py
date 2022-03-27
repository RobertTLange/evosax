import jax
import jax.numpy as jnp
import chex
from typing import Optional
from functools import partial
from .obs_norm import ObsNormalizer


class BraxFitness(object):
    def __init__(
        self,
        env_name: str = "ant",
        num_env_steps: int = 1000,
        num_rollouts: int = 16,
        legacy_spring: bool = True,
        test: bool = False,
        n_devices: Optional[int] = None,
    ):
        try:
            from brax import envs
        except ImportError:
            raise ImportError(
                "You need to install `brax` to use its fitness rollouts."
            )
        self.env_name = env_name
        self.num_env_steps = num_env_steps
        self.num_rollouts = num_rollouts
        self.steps_per_member = num_env_steps * num_rollouts
        self.test = test

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
        self.env = envs.create(
            env_name=self.env_name,
            episode_length=num_env_steps,
            legacy_spring=legacy_spring,
        )
        self.action_shape = self.env.action_size
        self.input_shape = (self.env.observation_size,)
        self.obs_normalizer = ObsNormalizer(self.input_shape, dummy=True)
        self.obs_params = self.obs_normalizer.get_init_params()
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

        # vmap over stochastic evaluations
        self.rollout_repeats = jax.vmap(self.single_rollout, in_axes=(0, None))
        self.rollout_pop = jax.vmap(
            self.rollout_repeats, in_axes=(None, map_dict)
        )
        # pmap over popmembers if > 1 device is available - otherwise pmap
        if self.n_devices > 1:
            self.rollout_map = self.rollout_pmap
            print(
                "BraxFitness: More than one device detected. Please make sure"
                " that the ES population size divides evenly across the number"
                " of devices to pmap/parallelize over."
            )
        else:
            self.rollout_map = self.rollout_pop

    def rollout_pmap(self, rng_input, policy_params):
        """Parallelize rollout across devices. Split keys/reshape correctly."""
        keys_pmap = jnp.tile(rng_input, (self.n_devices, 1, 1))
        rew_dev, obs_dev, masks_dev = jax.pmap(self.rollout_pop)(
            keys_pmap, policy_params
        )
        rew_re = rew_dev.reshape(-1, self.num_rollouts)
        obs_re = obs_dev.reshape(
            -1, self.num_rollouts, self.num_env_steps, self.env.observation_size
        )
        masks_re = masks_dev.reshape(
            -1, self.num_rollouts, self.num_env_steps, 1
        )
        return rew_re, obs_re, masks_re

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, rng_input, policy_params):
        """Placeholder fn call for rolling out a population for multi-evals."""
        rng_pop = jax.random.split(rng_input, self.num_rollouts)
        scores, all_obs, masks = self.rollout_map(rng_pop, policy_params)
        # Update normalization parameters if train case!
        if not self.test:
            obs_re = all_obs.reshape(
                -1, self.num_env_steps, self.input_shape[0]
            )
            obs_re = jnp.swapaxes(obs_re, 0, 1)
            masks_re = masks.reshape(-1, self.num_env_steps, 1)
            masks_re = jnp.swapaxes(masks_re, 0, 1)
            self.obs_params = self.obs_normalizer.update_normalization_params(
                obs_buffer=obs_re, obs_mask=masks_re, obs_params=self.obs_params
            )
        return scores

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
            org_obs = state.obs
            norm_obs = self.obs_normalizer.normalize_obs(
                org_obs, self.obs_params
            )
            action = self.network(
                {"params": policy_params}, norm_obs, rng=rng_net
            )
            next_s = self.env.step(state, action)
            carry = [next_s, policy_params, rng]
            return carry, [state.reward, state.done, org_obs]

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step, [state, policy_params, rng], (), self.num_env_steps
        )
        # Return masked sum of rewards accumulated by agent in episode
        rewards, dones, all_obs = scan_out[0], scan_out[1], scan_out[2]
        rewards = rewards.reshape(self.num_env_steps, 1)
        ep_mask = (jnp.cumsum(dones) < 1).reshape(self.num_env_steps, 1)
        return jnp.sum(rewards), all_obs, ep_mask

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
            org_obs = state.obs
            norm_obs = self.obs_normalizer.normalize_obs(
                state.obs, self.obs_params
            )
            hidden, action = self.network(
                {"params": policy_params}, norm_obs, hidden, rng_net
            )
            next_s = self.env.step(state, action)
            carry = [next_s, policy_params, rng, hidden]
            return carry, [state.reward, state.done, org_obs]

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step,
            [state, policy_params, rng, hidden],
            (),
            self.num_env_steps,
        )
        # Return masked sum of rewards accumulated by agent in episode
        rewards, dones, all_obs = scan_out[0], scan_out[1], scan_out[2]
        rewards = rewards.reshape(self.num_env_steps, 1)
        ep_mask = (jnp.cumsum(dones) < 1).reshape(self.num_env_steps, 1)
        return jnp.sum(rewards), all_obs, ep_mask
