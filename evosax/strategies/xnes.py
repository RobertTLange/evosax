import jax
import jax.numpy as jnp
from ..strategy import Strategy


class xNES(Strategy):
    def __init__(self, popsize: int, num_dims: int):
        """Exponential Natural ES (Wierstra et al., 2014).
        The code & example are heavily adopted from a scipy implementation:
        https://github.com/chanshing/xnes
        """
        super().__init__(num_dims, popsize)

    @property
    def default_params(self):
        """Return default parameters of evolutionary strategy."""
        params = {
            "eta_mean": 1.0,
            "use_adaptive_sampling": False,
            "use_fitness_shaping": True,
            "eta_sigma_init": 3
            * (3 + jnp.log(self.num_dims))
            * (1.0 / (5 * self.num_dims * jnp.sqrt(self.num_dims))),
            "eta_bmat": 3
            * (3 + jnp.log(self.num_dims))
            * (1.0 / (5 * self.num_dims * jnp.sqrt(self.num_dims))),
        }
        return params

    def initialize_strategy(self, rng, params):
        """`initialize` the evolutionary strategy."""
        amat = jnp.eye(self.num_dims)
        sigma = abs(jax.scipy.linalg.det(amat)) ** (1.0 / self.num_dims)
        bmat = amat * (1.0 / sigma)
        # Utility helper for fitness shaping - doesn't work without?!
        a = jnp.log(1 + 0.5 * self.popsize)
        utilities = jnp.array(
            [jnp.maximum(0, a - jnp.log(k)) for k in range(1, self.popsize + 1)]
        )
        utilities /= jnp.sum(utilities)
        utilities -= 1.0 / self.popsize  # broadcast
        utilities = utilities[::-1]  # ascending order

        state = {
            "mean": jnp.zeros(self.num_dims),
            "sigma": sigma,
            "sigma_old": sigma,
            "amat": amat,
            "bmat": bmat,
            "gen_counter": 0,
            "noise": jnp.zeros((self.popsize, self.num_dims)),
            "eta_sigma": params["eta_sigma_init"],
            "utilities": utilities,
        }

        return state

    def ask_strategy(self, rng, state, params):
        """`ask` for new parameter candidates to evaluate next."""
        noise = jax.random.normal(rng, (self.popsize, self.num_dims))
        x = state["mean"] + state["sigma"] * jnp.dot(noise, state["bmat"])
        state["noise"] = noise
        return x, state

    def tell_strategy(self, x, fitness, state, params):
        """`tell` performance data for strategy state update."""
        state["gen_counter"] = state["gen_counter"] + 1
        # By default the ES maximizes the objective
        fitness_re = -fitness
        isort = fitness_re.argsort()
        sorted_fitness = fitness_re[isort]
        sorted_noise = state["noise"][isort]
        sorted_candidates = x[isort]
        fitness_shaped = jax.lax.select(
            params["use_fitness_shaping"], state["utilities"], sorted_fitness
        )

        use_adasam = jnp.logical_and(
            params["use_adaptive_sampling"], state["gen_counter"] > 1
        )  # sigma_old must be available
        state["eta_sigma"] = jax.lax.select(
            use_adasam,
            self.adaptive_sampling(
                state["eta_sigma"],
                state["mean"],
                state["sigma"],
                state["bmat"],
                state["sigma_old"],
                sorted_candidates,
                params["eta_sigma_init"],
            ),
            state["eta_sigma"],
        )

        dj_delta = jnp.dot(fitness_shaped, sorted_noise)
        dj_mmat = jnp.dot(
            sorted_noise.T, sorted_noise * fitness_shaped.reshape(self.popsize, 1)
        ) - jnp.sum(fitness_shaped) * jnp.eye(self.num_dims)
        dj_sigma = jnp.trace(dj_mmat) * (1.0 / self.num_dims)
        dj_bmat = dj_mmat - dj_sigma * jnp.eye(self.num_dims)

        state["sigma_old"] = state["sigma"]
        state["mean"] += (
            params["eta_mean"] * state["sigma"] * jnp.dot(state["bmat"], dj_delta)
        )
        state["sigma"] = state["sigma_old"] * jnp.exp(
            0.5 * state["eta_sigma"] * dj_sigma
        )
        state["bmat"] = jnp.dot(
            state["bmat"], jax.scipy.linalg.expm(0.5 * params["eta_bmat"] * dj_bmat)
        )
        return state

    def adaptive_sampling(
        self, eta_sigma, mu, sigma, bmat, sigma_old, z_try, eta_sigma_init
    ):
        """Adaptation sampling."""
        c = 0.1
        rho = 0.5 - 1.0 / (3 * (self.num_dims + 1))  # empirical

        bbmat = jnp.dot(bmat.T, bmat)
        cov = sigma ** 2 * bbmat
        sigma_ = sigma * jnp.sqrt(sigma * (1.0 / sigma_old))  # increase by 1.5
        cov_ = sigma_ ** 2 * bbmat

        p0 = jax.scipy.stats.multivariate_normal.logpdf(z_try, mean=mu, cov=cov)
        p1 = jax.scipy.stats.multivariate_normal.logpdf(z_try, mean=mu, cov=cov_)
        w = jnp.exp(p1 - p0)

        # Mann-Whitney. It is assumed z_try was in ascending order.
        n_ = jnp.sum(w)
        u_ = jnp.sum(w * (jnp.arange(self.popsize) + 0.5))

        u_mu = self.popsize * n_ * 0.5
        u_sigma = jnp.sqrt(self.popsize * n_ * (self.popsize + n_ + 1) / 12.0)
        cum = jax.scipy.stats.norm.cdf(u_, loc=u_mu, scale=u_sigma)

        decrease = cum < rho
        eta_out = jax.lax.select(
            decrease,
            (1 - c) * eta_sigma + c * eta_sigma_init,
            jnp.minimum(1, (1 + c) * eta_sigma),
        )
        return eta_out


if __name__ == "__main__":

    def f(x):  # sin(x^2+y^2)/(x^2+y^2)
        r = jnp.sum(x ** 2)
        return -jnp.sin(r) / r

    batch_func = jax.vmap(f, in_axes=0)

    rng = jax.random.PRNGKey(0)
    strategy = xNES(popsize=50, num_dims=2)
    params = strategy.default_params
    params["use_adaptive_sampling"] = True
    params["use_fitness_shaping"] = True
    params["eta_bmat"] = 0.01
    params["eta_sigma"] = 0.1

    state = strategy.initialize(rng, params)
    state["mean"] = jnp.array([9999.0, -9999.0])  # a bad init guess
    fitness_log = []
    num_iters = 5000
    for t in range(num_iters):
        rng, rng_iter = jax.random.split(rng)
        y, state = strategy.ask(rng_iter, state, params)
        fitness = batch_func(y)
        state = strategy.tell(y, fitness, state, params)
        best_id = jnp.argmin(fitness)
        fitness_log.append(fitness[best_id])
        if t % 500 == 0:
            print(t, jnp.min(jnp.array(fitness_log)), state["mean"])
