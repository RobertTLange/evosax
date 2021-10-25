import jax
import jax.numpy as jnp
from functools import partial
from ..strategy import Strategy


def outer_adam_step(state, params, grads):
    state["m"] = (1 - params["beta_1"]) * grads + params["beta_1"] * state["m"]
    state["v"] = (1 - params["beta_2"]) * (grads ** 2) + params["beta_2"] * state["v"]
    mhat = state["m"] / (1 - params["beta_1"] ** (state["outer_step_counter"] + 1))
    vhat = state["v"] / (1 - params["beta_2"] ** (state["outer_step_counter"] + 1))
    state["mean"] -= params["lrate"] * mhat / (jnp.sqrt(vhat) + params["eps"])
    return state


class Persistent_ES(Strategy):
    def __init__(self, popsize: int, num_dims: int):
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
        }

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, rng, params) -> dict:
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
            "outer_step_counter": 0,
            "inner_step_counter": 0,
        }
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng, state, params):
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

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, x, fitness, state, params):
        """`tell` update to ES state."""
        theta_grad = jnp.mean(
            state["pert_accum"] * fitness.reshape(-1, 1) / (state["sigma"] ** 2), axis=0
        )
        state = outer_adam_step(state, params, theta_grad)
        state["outer_step_counter"] += 1
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


if __name__ == "__main__":

    def loss(x):
        """Inner loss."""
        return (
            jnp.sqrt(x[0] ** 2 + 5)
            - jnp.sqrt(5)
            + jnp.sin(x[1]) ** 2 * jnp.exp(-5 * x[0] ** 2)
            + 0.25 * jnp.abs(x[1] - 100)
        )

    # Gradient of inner loss
    loss_grad = jax.grad(loss)

    def update(state, i):
        """Performs a single inner problem update, e.g., a single unroll step."""
        (L, x, theta, t_curr, T, K) = state
        lr = jnp.exp(theta[0]) * (T - t_curr) / T + jnp.exp(theta[1]) * t_curr / T
        x = x - lr * loss_grad(x)
        L += loss(x) * (t_curr < T)
        t_curr += 1
        return (L, x, theta, t_curr, T, K), x

    @partial(jax.jit, static_argnums=(3, 4))
    def unroll(x_init, theta, t0, T, K):
        """Unroll the inner problem for K steps."""
        L = 0.0
        initial_state = (L, x_init, theta, t0, T, K)
        state, outputs = jax.lax.scan(update, initial_state, None, length=K)
        (L, x_curr, theta, t_curr, T, K) = state
        return L, x_curr

    strategy = Persistent_ES(popsize=100, num_dims=2)
    params = strategy.default_params
    rng = jax.random.PRNGKey(5)
    state = strategy.initialize(rng, params)

    # Initialize inner parameters
    t = 0
    theta = jnp.log(jnp.array([0.01, 0.01]))
    x = jnp.array([1.0, 1.0])
    xs = jnp.ones((N, 2)) * jnp.array([1.0, 1.0])
    popsize = 100

    for i in range(10000):
        rng, skey = jax.random.split(rng)
        if t >= params["T"]:
            # Reset the inner problem: iteration, parameters
            t = 0
            xs = jnp.ones((popsize, 2)) * jnp.array([1.0, 1.0])
            x = jnp.array([1.0, 1.0])
        theta_gen, state = strategy.ask(rng, state, params)

        # Unroll inner problem for K steps using antithetic perturbations
        L, xs = jax.vmap(unroll, in_axes=(0, 0, None, None, None))(
            xs, theta_gen, t, params["T"], params["K"]
        )

        state = strategy.tell(theta_gen, L, state, params)
        t += params["K"]

        # Teset evaluation!
        if i % 1000 == 0:
            L, _ = unroll(
                jnp.array([1.0, 1.0]), state["mean"], 0, params["T"], params["T"]
            )
            print(i, state["mean"], L)
