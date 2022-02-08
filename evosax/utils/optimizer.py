import jax
import jax.numpy as jnp


class SGD_Optimizer(object):
    def __init__(self, num_dims: int):
        self.opt_name = "sgd"
        self.num_dims = num_dims

    @property
    def default_params(self) -> dict:
        """Return default SGD+Momentum parameters."""
        return {
            "lrate_init": 0.01,
            "lrate_decay": 0.999,
            "lrate_limit": 0.001,
            "momentum": 0.9,
        }

    def initialize(self, params):
        """Initialize the momentum trace of the optimizer."""
        return {"m": jnp.zeros(self.num_dims), "lrate": params["lrate_init"]}

    def step(self, grads, state, params):
        """Perform a simple SGD + Momentum step."""
        state["m"] = grads + params["momentum"] * state["m"]
        state["mean"] -= state["lrate"] * state["m"]
        return state

    def update(self, state, params):
        """Exponentially the learning rate if desired."""
        state["lrate"] *= params["lrate_decay"]
        state["lrate"] = jnp.maximum(state["lrate"], params["lrate_limit"])
        return state


class Adam_Optimizer(object):
    def __init__(self, num_dims: int):
        self.opt_name = "adam"
        self.num_dims = num_dims

    @property
    def default_params(self):
        """Return default SGD+Momentum parameters."""
        return {
            "lrate_init": 0.01,
            "lrate_decay": 0.999,
            "lrate_limit": 0.001,
            "beta_1": 0.99,  # beta_1 outer step
            "beta_2": 0.999,  # beta_2 outer step
            "eps": 1e-8,  # eps constant outer step,
        }

    def initialize(self, params):
        """Initialize the momentum trace of the optimizer."""
        return {
            "m": jnp.zeros(self.num_dims),
            "v": jnp.zeros(self.num_dims),
            "lrate": params["lrate_init"],
        }

    def step(self, grads, state, params):
        """Perform a simple Adam GD step (Kingma & Ba, 2015)."""
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

    def update(self, state, params):
        """Exponentially the learning rate if desired."""
        state["lrate"] *= params["lrate_decay"]
        state["lrate"] = jnp.maximum(state["lrate"], params["lrate_limit"])
        return state


class RMSProp_Optimizer(object):
    def __init__(self, num_dims: int):
        self.opt_name = "rmsprop"
        self.num_dims = num_dims

    @property
    def default_params(self):
        """Return default SGD+Momentum parameters."""
        return {
            "lrate_init": 0.01,
            "lrate_decay": 0.999,
            "lrate_limit": 0.001,
            "momentum": 0.9,
            "beta": 0.99,
            "eps": 1e-8,  # eps constant outer step,
        }

    def initialize(self, params):
        """Initialize the momentum trace of the optimizer."""
        return {
            "m": jnp.zeros(self.num_dims),
            "v": jnp.zeros(self.num_dims),
            "lrate": params["lrate_init"],
        }

    def step(self, grads, state, params):
        """Perform a simple RMSprop GD step (Hinton lecture).
        https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf"""
        state["v"] = (1 - params["beta"]) * (grads ** 2) + params[
            "beta"
        ] * state["v"]
        state["m"] = params["momentum"] * state["m"] + grads / (
            jnp.sqrt(state["v"]) + params["eps"]
        )
        state["mean"] -= state["lrate"] * state["m"]
        return state

    def update(self, state, params):
        """Exponentially the learning rate if desired."""
        state["lrate"] *= params["lrate_decay"]
        state["lrate"] = jnp.maximum(state["lrate"], params["lrate_limit"])
        return state


class ClipUp_Optimizer(object):
    def __init__(self, num_dims: int):
        self.opt_name = "clipup"
        self.num_dims = num_dims

    @property
    def default_params(self):
        """Return default SGD+Momentum parameters."""
        return {
            "lrate_init": 0.15,
            "lrate_decay": 0.999,
            "lrate_limit": 0.05,
            "max_speed": 0.3,
            "momentum": 0.9,
        }

    def initialize(self, params):
        """Initialize the momentum trace of the optimizer."""
        return {
            "m": jnp.zeros(self.num_dims),
            "lrate": params["lrate_init"],
        }

    def step(self, grads, state, params):
        """Perform a ClipUp step (Toklu et al., 2020).
        Heuristics: momentum = 0.9, lrate = vmax/2, vmax = small"""

        # Normalize length of gradients - vmax & alpha control max step magnitude
        grad_magnitude = jnp.sqrt(jnp.sum(grads * grads))
        gradient = grads / grad_magnitude
        step = gradient * state["lrate"]
        velocity = params["momentum"] * state["m"] + step

        def clip(velocity: jnp.ndarray, max_speed: float):
            """Rescale clipped velocities."""
            vel_magnitude = jnp.sqrt(jnp.sum(velocity * velocity))
            ratio_scale = vel_magnitude > max_speed
            scaled_vel = velocity * (max_speed / vel_magnitude)
            x_out = jax.lax.select(ratio_scale, scaled_vel, velocity)
            return x_out

        # Clip the update velocity and apply the update
        state["m"] = clip(velocity, params["max_speed"])
        state["mean"] -= state["lrate"] * state["m"]
        return state

    def update(self, state, params):
        """Exponentially the learning rate if desired."""
        state["lrate"] *= params["lrate_decay"]
        state["lrate"] = jnp.maximum(state["lrate"], params["lrate_limit"])
        return state
