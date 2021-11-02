import jax.numpy as jnp


def adam_step(state, params, grads):
    state["m"] = (1 - params["beta_1"]) * grads + params["beta_1"] * state["m"]
    state["v"] = (1 - params["beta_2"]) * (grads ** 2) + params["beta_2"] * state["v"]
    mhat = state["m"] / (1 - params["beta_1"] ** (state["gen_counter"] + 1))
    vhat = state["v"] / (1 - params["beta_2"] ** (state["gen_counter"] + 1))
    state["mean"] -= params["lrate"] * mhat / (jnp.sqrt(vhat) + params["eps"])
    return state
