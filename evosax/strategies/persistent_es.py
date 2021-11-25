import jax
import jax.numpy as jnp
from ..strategy import Strategy
from ..utils import adam_step


class Persistent_ES(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Persistent ES (Vicol et al., 2021).
        The code & example are heavily adopted from the supplementary material:
        http://proceedings.mlr.press/v139/vicol21a/vicol21a-supp.pdf
        """
        super().__init__(num_dims, popsize)
        assert not self.popsize & 1, "Population size must be even"

    @property
    def default_params(self) -> dict:
        return {
            "lrate": 5e-3,  # Adam learning rate outer step
            "beta_1": 0.99,  # beta_1 outer step
            "beta_2": 0.999,  # beta_2 outer step
            "eps": 1e-8,  # eps constant outer step
            "sigma_init": 0.1,  # Perturbation Std
            "T": 100,  # Total inner problem length
            "K": 10,  # Truncation length for partial unrolls
            "init_min": 0,
            "init_max": 0,
            "clip_min": -jnp.finfo(jnp.float32).max,
            "clip_max": jnp.finfo(jnp.float32).max,
        }

    def initialize_strategy(self, rng, params) -> dict:
        """`initialize` the differential evolution strategy."""
        state = {
            "mean": jax.random.uniform(
                rng,
                (self.num_dims,),
                minval=params["init_min"],
                maxval=params["init_max"],
            ),
            "m": jnp.zeros(self.num_dims),
            "v": jnp.zeros(self.num_dims),
            "pert_accum": jnp.zeros((self.popsize, self.num_dims)),
            "sigma": params["sigma_init"],
            "inner_step_counter": 0,
        }
        return state

    def ask_strategy(self, rng, state, params):
        """`ask` for new proposed candidates to evaluate next."""
        # Generate antithetic perturbations
        pos_perts = (
            jax.random.normal(rng, (self.popsize // 2, self.num_dims)) * state["sigma"]
        )
        neg_perts = -pos_perts
        perts = jnp.concatenate([pos_perts, neg_perts], axis=0)
        # Add the perturbations from this unroll to the perturbation accumulators
        state["pert_accum"] += perts
        y = state["mean"] + perts
        return jnp.squeeze(y), state

    def tell_strategy(self, x, fitness, state, params):
        """`tell` update to ES state."""
        theta_grad = jnp.mean(
            state["pert_accum"] * fitness.reshape(-1, 1) / (state["sigma"] ** 2), axis=0
        )
        state = adam_step(state, params, theta_grad)
        state["inner_step_counter"] += params["K"]

        # Reset accumulated antithetic noise memory if done with inner problem
        reset = state["inner_step_counter"] >= params["T"]
        state["inner_step_counter"] = jax.lax.select(
            reset, 0, state["inner_step_counter"]
        )
        state["pert_accum"] = jax.lax.select(
            reset, jnp.zeros((self.popsize, self.num_dims)), state["pert_accum"]
        )
        return state
