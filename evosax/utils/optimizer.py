import jax
import jax.numpy as jnp
import chex


class Optimizer(object):
    def __init__(self, num_dims: int):
        """Simple JAX-Compatible Optimizer Class."""
        self.num_dims = num_dims

    @property
    def default_params(self) -> chex.ArrayTree:
        """Return shared and optimizer-specific default parameters."""
        params_shared = {
            "lrate_init": 0.01,
            "lrate_decay": 0.999,
            "lrate_limit": 0.001,
            # TODO: Add gradient clipping - select leads to more compute
            # "use_clip_by_global_norm": False,
            # "clip_global_norm": 5,
            # "use_clip_by_value": False,
            # "clip_value": 5,
        }
        return {**params_shared, **self.params_opt}

    def initialize(self, params: chex.ArrayTree) -> chex.ArrayTree:
        """Initialize the optimizer state."""
        return self.initialize_opt(params)

    def step(
        self, grads: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Perform a gradient-based update step."""
        return self.step_opt(grads, state, params)

    def update(
        self, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Exponentially decay the learning rate if desired."""
        state["lrate"] *= params["lrate_decay"]
        state["lrate"] = jnp.maximum(state["lrate"], params["lrate_limit"])
        return state

    @property
    def params_opt(self) -> chex.ArrayTree:
        """Optimizer-specific hyperparameters."""
        raise NotImplementedError

    def initialize_opt(self, params: chex.ArrayTree) -> chex.ArrayTree:
        """Optimizer-specific initialization of optimizer state."""
        raise NotImplementedError

    def step_opt(
        self, grads: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Optimizer-specific step to update parameter estimates."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, num_dims: int):
        """Simple JAX-Compatible SGD + Momentum optimizer."""
        super().__init__(num_dims)
        self.opt_name = "sgd"

    @property
    def params_opt(self) -> chex.ArrayTree:
        """Return default SGD+Momentum parameters."""
        return {
            "momentum": 0.9,
        }

    def initialize_opt(self, params: chex.ArrayTree) -> chex.ArrayTree:
        """Initialize the momentum trace of the optimizer."""
        return {"m": jnp.zeros(self.num_dims), "lrate": params["lrate_init"]}

    def step_opt(
        self, grads: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Perform a simple SGD + Momentum step."""
        state["m"] = grads + params["momentum"] * state["m"]
        state["mean"] -= state["lrate"] * state["m"]
        return state


class Adam(Optimizer):
    def __init__(self, num_dims: int):
        """JAX-Compatible Adam Optimizer (Kingma & Ba, 2015)
        Reference: https://arxiv.org/abs/1412.6980"""
        super().__init__(num_dims)
        self.opt_name = "adam"

    @property
    def params_opt(self) -> chex.ArrayTree:
        """Return default Adam parameters."""
        return {
            "beta_1": 0.99,
            "beta_2": 0.999,
            "eps": 1e-8,
        }

    def initialize_opt(self, params: chex.ArrayTree) -> chex.ArrayTree:
        """Initialize the m, v trace of the optimizer."""
        return {
            "m": jnp.zeros(self.num_dims),
            "v": jnp.zeros(self.num_dims),
            "lrate": params["lrate_init"],
        }

    def step_opt(
        self, grads: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Perform a simple Adam GD step."""
        state["m"] = (1 - params["beta_1"]) * grads + params["beta_1"] * state[
            "m"
        ]
        state["v"] = (1 - params["beta_2"]) * (grads ** 2) + params[
            "beta_2"
        ] * state["v"]
        mhat = state["m"] / (1 - params["beta_1"] ** (state["gen_counter"] + 1))
        vhat = state["v"] / (1 - params["beta_2"] ** (state["gen_counter"] + 1))
        state["mean"] -= (
            state["lrate"] * mhat / (jnp.sqrt(vhat) + params["eps"])
        )
        return state


class RMSProp(Optimizer):
    def __init__(self, num_dims: int):
        """JAX-Compatible RMSProp Optimizer (Hinton et al., 2012)
        Reference: https://tinyurl.com/2sbbcnrv"""
        super().__init__(num_dims)
        self.opt_name = "rmsprop"

    @property
    def params_opt(self) -> chex.ArrayTree:
        """Return default RMSProp parameters."""
        return {
            "momentum": 0.9,
            "beta": 0.99,
            "eps": 1e-8,
        }

    def initialize_opt(self, params: chex.ArrayTree) -> chex.ArrayTree:
        """Initialize the m, v trace of the optimizer."""
        return {
            "m": jnp.zeros(self.num_dims),
            "v": jnp.zeros(self.num_dims),
            "lrate": params["lrate_init"],
        }

    def step_opt(
        self, grads: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Perform a simple RMSprop GD step."""
        state["v"] = (1 - params["beta"]) * (grads ** 2) + params[
            "beta"
        ] * state["v"]
        state["m"] = params["momentum"] * state["m"] + grads / (
            jnp.sqrt(state["v"]) + params["eps"]
        )
        state["mean"] -= state["lrate"] * state["m"]
        return state


class ClipUp(Optimizer):
    def __init__(self, num_dims: int):
        """JAX-Compatible ClipUp Optimizer (Toklu et al., 2020)
        Reference: https://arxiv.org/abs/2008.02387"""
        super().__init__(num_dims)
        self.opt_name = "clipup"

    @property
    def params_opt(self) -> chex.ArrayTree:
        """Return default ClipUp parameters."""
        return {
            "lrate_init": 0.15,
            "lrate_decay": 0.999,
            "lrate_limit": 0.05,
            "max_speed": 0.3,
            "momentum": 0.9,
        }

    def initialize_opt(self, params: chex.ArrayTree) -> chex.ArrayTree:
        """Initialize the momentum trace of the optimizer."""
        return {
            "m": jnp.zeros(self.num_dims),
            "lrate": params["lrate_init"],
        }

    def step_opt(
        self, grads: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Perform a ClipUp step. mom = 0.9, lrate = vmax/2, vmax = small"""
        # Normalize length of gradients - vmax & alpha control max step magnitude
        grad_magnitude = jnp.sqrt(jnp.sum(grads * grads))
        gradient = grads / (grad_magnitude + 1e-08)
        step = gradient * state["lrate"]
        velocity = params["momentum"] * state["m"] + step

        def clip(velocity: chex.Array, max_speed: float):
            """Rescale clipped velocities."""
            vel_magnitude = jnp.sqrt(jnp.sum(velocity * velocity))
            ratio_scale = vel_magnitude > max_speed
            scaled_vel = velocity * (max_speed / (vel_magnitude + 1e-08))
            x_out = jax.lax.select(ratio_scale, scaled_vel, velocity)
            return x_out

        # Clip the update velocity and apply the update
        state["m"] = clip(velocity, params["max_speed"])
        state["mean"] -= state["lrate"] * state["m"]
        return state
