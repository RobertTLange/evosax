import chex
import jax
import jax.numpy as jnp


class GymnaxProblem:
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        num_env_steps: int | None = None,
        num_rollouts: int = 16,
        env_kwargs: dict = {},
        env_params: dict = {},
    ):
        self.env_name = env_name
        self.num_rollouts = num_rollouts

        try:
            import gymnax
        except ImportError:
            raise ImportError(
                "You need to install `gymnax` to use its fitness rollouts."
            )

        # Define the RL environment & replace default parameters if desired
        self.env, self.env_params = gymnax.make(env_name, **env_kwargs)
        self.env_params.replace(**env_params)

        if num_env_steps is None:
            self.num_env_steps = self.env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps
        self.steps_per_member = self.num_env_steps * num_rollouts

        self.action_shape = self.env.num_actions
        self.input_shape = self.env.observation_space(self.env_params).shape

        # Keep track of total steps executed in environment
        self.total_env_steps = 0

    def set_apply_fn(self, network_apply, carry_init=None):
        """Set the network forward function."""
        self.network = network_apply
        # Set rollout function based on model architecture
        if carry_init is not None:
            self.single_rollout = self.rollout_rnn
            self.carry_init = carry_init
        else:
            self.single_rollout = self.rollout_ffw
        self.rollout_repeats = jax.vmap(self.single_rollout, in_axes=(0, None))
        self.rollout_pop = jax.vmap(self.rollout_repeats, in_axes=(None, 0))

    def eval(self, key: jax.Array, policy_params: chex.ArrayTree):
        """Evaluate a population of policies."""
        keys = jax.random.split(key, self.num_rollouts)
        scores, masks = jax.jit(self.rollout_pop)(keys, policy_params)
        # Update total step counter using only transitions before termination
        self.total_env_steps += masks.sum()
        return scores

    def rollout_ffw(self, key: jax.Array, policy_params: chex.ArrayTree):
        """Rollout an episode with lax.scan."""
        key_reset, key_step = jax.random.split(key)

        obs, state = self.env.reset(key_reset, self.env_params)

        def policy_step(carry, _):
            obs, state, policy_params, key, cum_reward, valid_mask = carry

            key, key_action, key_step = jax.random.split(key, 3)
            action = self.network(policy_params, obs, key=key_action)
            next_o, next_s, reward, done, _ = self.env.step(
                key_step, state, action, self.env_params
            )

            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = (
                next_o.squeeze(),
                next_s,
                policy_params,
                key,
                new_cum_reward,
                new_valid_mask,
            )
            return carry, (new_valid_mask,)

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            (
                obs,
                state,
                policy_params,
                key_step,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ),
            length=self.num_env_steps,
        )

        # Return the sum of rewards accumulated by agent in episode rollout
        ep_mask = scan_out
        cum_return = carry_out[-2].squeeze()
        return cum_return, jnp.array(ep_mask)

    def rollout_rnn(self, key: jax.Array, policy_params: chex.ArrayTree):
        """Rollout a jitted episode with lax.scan."""
        key_reset, key_step = jax.random.split(key)

        obs, state = self.env.reset(key_reset, self.env_params)
        hidden = self.carry_init()

        def policy_step(carry, _):
            obs, state, policy_params, key, hidden, cum_reward, valid_mask = carry

            key, key_action, key_step = jax.random.split(key, 3)
            hidden, action = self.network(policy_params, obs, hidden, key=key_action)
            next_o, next_s, reward, done, _ = self.env.step(
                key_step, state, action, self.env_params
            )

            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = (
                next_o.squeeze(),
                next_s,
                policy_params,
                key,
                hidden,
                new_cum_reward,
                new_valid_mask,
            )
            return carry, (new_valid_mask,)

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            (
                obs,
                state,
                policy_params,
                key_step,
                hidden,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ),
            length=self.num_env_steps,
        )

        # Return masked sum of rewards accumulated by agent in episode
        ep_mask = scan_out
        cum_return = carry_out[-2].squeeze()
        return cum_return, jnp.array(ep_mask)
