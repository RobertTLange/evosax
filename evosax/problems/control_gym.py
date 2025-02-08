import chex
import jax
import jax.numpy as jnp


class GymnaxFitness:
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

    def rollout(self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree):
        """Placeholder fn call for rolling out a population for multi-evals."""
        rng_pop = jax.random.split(rng_input, self.num_rollouts)
        scores, masks = jax.jit(self.rollout_pop)(rng_pop, policy_params)
        # Update total step counter using only transitions before termination
        self.total_env_steps += masks.sum()
        return scores

    def rollout_ffw(self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action = self.network(policy_params, obs, rng=rng_net)
            next_o, next_s, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_o.squeeze(),
                next_s,
                policy_params,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [new_valid_mask]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_params,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        ep_mask = scan_out
        cum_return = carry_out[-2].squeeze()
        return cum_return, jnp.array(ep_mask)

    def rollout_rnn(self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree):
        """Rollout a jitted episode with lax.scan."""
        # Reset the environment
        rng, rng_reset = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)
        hidden = self.carry_init()

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            (
                obs,
                state,
                policy_params,
                rng,
                hidden,
                cum_reward,
                valid_mask,
            ) = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            hidden, action = self.network(policy_params, obs, hidden, rng_net)
            next_o, next_s, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry, y = (
                [
                    next_o.squeeze(),
                    next_s,
                    policy_params,
                    rng,
                    hidden,
                    new_cum_reward,
                    new_valid_mask,
                ],
                [new_valid_mask],
            )
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_params,
                rng,
                hidden,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps,
        )
        # Return masked sum of rewards accumulated by agent in episode
        ep_mask = scan_out
        cum_return = carry_out[-2].squeeze()
        return cum_return, jnp.array(ep_mask)
