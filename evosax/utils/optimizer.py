import jax
import jax.numpy as jnp


def adam_step(state, params, grads):
    """Perform a simple Adam GD step (Kingma & Ba, 2015)."""
    state["m"] = (1 - params["beta_1"]) * grads + params["beta_1"] * state["m"]
    state["v"] = (1 - params["beta_2"]) * (grads ** 2) + params["beta_2"] * state["v"]
    mhat = state["m"] / (1 - params["beta_1"] ** (state["gen_counter"] + 1))
    vhat = state["v"] / (1 - params["beta_2"] ** (state["gen_counter"] + 1))
    state["mean"] -= params["lrate"] * mhat / (jnp.sqrt(vhat) + params["eps"])
    return state


def clipup_step(state, params, grads):
    """Perform a ClipUp step (Toklu et al., 2020)."""
    grad_magnitude = jnp.sqrt(jnp.sum(grads * grads))
    gradient = grads / grad_magnitude
    step = gradient * params["lrate"]
    velocity = params["momentum"] * state["velocity"] + step
    state["velocity"] = self.clip(velocity, params["max_speed"])
    return state


def clip(velocity, max_speed: float):
    """Rescale clipped velocities."""
    vel_magnitude = jnp.sqrt(jnp.sum(velocity * velocity))
    ratio_scale = vel_magnitude > max_speed
    scaled_vel = velocity * (max_speed / vel_magnitude)
    x_out = jax.lax.select(ratio_scale, scaled_vel, velocity)
    return x_out
