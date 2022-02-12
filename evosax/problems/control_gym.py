import jax
import jax.numpy as jnp
import gymnax


class GymnaxFitness(object):
    def __init__(self, env_name: str = "CartPole-v1", num_env_steps: int = 200):
        self.env_name = env_name
        self.num_env_steps = num_env_steps

        # Define the RL environment & network forward fucntion
        self.env, self.env_params = gymnax.make(self.env_name)
        self.action_shape = self.env.action_space.n

    def set_apply_fn(self, network, recurrent: bool = False):
        """Set the network forward function."""
        self.network = network
        # Set rollout function based on model architecture
        if recurrent:
            self.rollout = self.rollout_rnn
        else:
            self.rollout = self.rollout_ffw

    def rollout_ffw(self, rng_input, policy_params):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action = self.network.apply(
                {"params": policy_params}, obs, rng=rng_net
            )
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

    def rollout_rnn(self, rng_input, policy_params):
        """Rollout a jitted gymnax episode with lax.scan."""
        # Reset the environment
        rng, rng_reset = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)
        hidden = self.network.initialize_carry()

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng, hidden = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            hidden, action = self.network.apply(
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

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape
