from .strategy import Strategy
import jax
from functools import partial
import jax.numpy as jnp


class CMA_ES(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.5):
        super().__init__(num_dims, popsize)
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

    @property
    def default_params(self):
        weights_prime = jnp.array(
            [jnp.log((self.popsize + 1) / 2) - jnp.log(i + 1)
             for i in range(self.popsize)])
        mu_eff = ((jnp.sum(weights_prime[:self.elite_popsize]) ** 2) /
                   jnp.sum(weights_prime[:self.elite_popsize] ** 2))
        mu_eff_minus = ((jnp.sum(weights_prime[self.elite_popsize:]) ** 2) /
                         jnp.sum(weights_prime[self.elite_popsize:] ** 2))

        # lrates for rank-one and rank-μ C updates
        alpha_cov = 2
        c_1 = alpha_cov / ((self.num_dims + 1.3) ** 2 + mu_eff)
        c_mu = jnp.minimum(1 - c_1 - 1e-8, alpha_cov * (mu_eff - 2 + 1 / mu_eff)
                  / ((self.num_dims + 2) ** 2 + alpha_cov * mu_eff / 2))
        min_alpha = min(1 + c_1 / c_mu,
                        1 + (2 * mu_eff_minus) / (mu_eff + 2),
                        (1 - c_1 - c_mu) / (self.num_dims * c_mu))
        positive_sum = jnp.sum(weights_prime[weights_prime > 0])
        negative_sum = jnp.sum(jnp.abs(weights_prime[weights_prime < 0]))
        weights = jnp.where(weights_prime >= 0,
                            1 / positive_sum * weights_prime,
                            min_alpha / negative_sum * weights_prime,)
        weights_truncated = jax.ops.index_update(
                                weights,
                                jax.ops.index[self.elite_popsize:], 0)

        # lrate for cumulation of step-size control and rank-one update
        c_sigma = (mu_eff + 2) / (self.num_dims + mu_eff + 5)
        d_sigma = 1 + 2 * jnp.maximum(0, jnp.sqrt((mu_eff - 1) / (self.num_dims + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.num_dims) / (self.num_dims + 4 + 2 * mu_eff / self.num_dims)
        chi_n = jnp.sqrt(self.num_dims) * (
            1.0 - (1.0 / (4.0 * self.num_dims)) + 1.0 / (21.0 * (self.num_dims ** 2)))

        params = {"mu_eff": mu_eff,
                  "c_1": c_1,
                  "c_mu": c_mu,
                  "c_m": 1,
                  "c_sigma": c_sigma,
                  "d_sigma": d_sigma,
                  "c_c": c_c,
                  "chi_n": chi_n,
                  "weights": weights,
                  "sigma_init": 1,
                  "weights_truncated": weights_truncated}
        return params

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, rng, params):
        """`initialize` the evolution strategy."""
        # Initialize evolution paths & covariance matrix
        state = {"p_sigma": jnp.zeros(self.num_dims),
                 "p_c": jnp.zeros(self.num_dims),
                 "sigma": params["sigma_init"],
                 "mean": jnp.zeros(self.num_dims),
                 "C": jnp.eye(self.num_dims),
                 "D": None,
                 "B": None,
                 "gen_counter": 0}
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng, state, params):
        """`ask` for new parameter candidates to evaluate next."""
        C, B, D = eigen_decomposition(state["C"], state["B"], state["D"])
        x = sample(rng, state["mean"], state["sigma"], B, D,
                   self.num_dims, self.popsize)
        state["C"], state["B"], state["D"] = C, B, D
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, x, fitness, state, params):
        """`tell` performance data for strategy state update."""
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        y_k, y_w, mean = update_mean(state["mean"],
                                     state["sigma"],
                                     sorted_solutions,
                                     params)

        p_sigma, C_2, C, B, D = update_p_sigma(state["C"],
                                               state["B"],
                                               state["D"],
                                               state["p_sigma"],
                                               y_w,
                                               params)

        p_c, norm_p_sigma, h_sigma = update_p_c(mean,
                                                p_sigma,
                                                state["p_c"],
                                                state["gen_counter"],
                                                y_w,
                                                params)

        C = update_covariance(mean, p_c, C, y_k, h_sigma, C_2, params)
        sigma = update_sigma(state["sigma"], norm_p_sigma, params)

        state["mean"] = mean
        state["p_sigma"] = p_sigma
        state["C"] = C
        state["B"] = B
        state["D"] = D
        state["p_c"] = p_c
        state["C"] = C
        state["sigma"] = sigma
        state["gen_counter"] += 1
        return state


def update_mean(mean, sigma, sorted_solutions, params):
    """ Update mean of strategy. """
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = (x_k - mean) / sigma  # ~ N(0, C)
    y_w = jnp.sum(y_k.T * params["weights_truncated"], axis=1)
    mean += params["c_m"] * sigma * y_w
    return y_k, y_w, mean


def update_p_sigma(C, B, D, p_sigma, y_w, params):
    """ Update evolution path for covariance matrix. """
    C, B, D = eigen_decomposition(C, B, D)
    C_2 = B.dot(jnp.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
    p_sigma_new = (1 - params["c_sigma"]) * p_sigma + jnp.sqrt(
        params["c_sigma"] * (2 - params["c_sigma"]) *
        params["mu_eff"]) * C_2.dot(y_w)
    _B, _D = None, None
    return p_sigma_new, C_2, C, _B, _D


def update_p_c(mean, p_sigma, p_c, gen_counter, y_w, params):
    """ Update evolution path for sigma/stepsize. """
    norm_p_sigma = jnp.linalg.norm(p_sigma)
    h_sigma_cond_left = norm_p_sigma / jnp.sqrt(
        1 - (1 - params["c_sigma"]) ** (2 * (gen_counter + 1)))
    h_sigma_cond_right = (1.4 + 2 / (mean.shape[0] + 1)) * params["chi_n"]
    h_sigma = 1.0 * (h_sigma_cond_left < h_sigma_cond_right)
    p_c_new = (1 - params["c_c"]) * p_c + h_sigma * jnp.sqrt(
               params["c_c"] * (2 - params["c_c"]) * params["mu_eff"]) * y_w
    return p_c_new, norm_p_sigma, h_sigma


def update_covariance(mean, p_c, C, y_k, h_sigma, C_2, params):
    """ Update cov. matrix estimator using rank 1 + μ updates. """
    w_io = params["weights"] * jnp.where(params["weights"] >= 0, 1,
                                         mean.shape[0]/
            (jnp.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + 1e-20))
    delta_h_sigma = (1 - h_sigma) * params["c_c"] * (2 - params["c_c"])
    rank_one = jnp.outer(p_c, p_c)
    rank_mu = jnp.sum(
        jnp.array([w * jnp.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0)
    C = ((1 + params["c_1"] * delta_h_sigma - params["c_1"]
          - params["c_mu"] * jnp.sum(params["weights"])) * C
         + params["c_1"] * rank_one + params["c_mu"] * rank_mu)
    return C


def update_sigma(sigma, norm_p_sigma, params):
    """ Update stepsize sigma. """
    sigma_new = (sigma * jnp.exp((params["c_sigma"] / params["d_sigma"])
                                 * (norm_p_sigma / params["chi_n"] - 1)))
    return sigma_new


def sample(rng, mean, sigma, B, D, n_dim, pop_size):
    """ Jittable Gaussian Sample Helper. """
    z = jax.random.normal(rng, (n_dim, pop_size)) # ~ N(0, I)
    y = B.dot(jnp.diag(D)).dot(z)               # ~ N(0, C)
    y = jnp.swapaxes(y, 1, 0)
    x = mean + sigma * y    # ~ N(m, σ^2 C)
    return x


def eigen_decomposition(C, B, D):
    """ Perform eigendecomposition of covariance matrix. """
    if B is not None and D is not None:
        return C, B, D
    C = (C + C.T) / 2
    D2, B = jnp.linalg.eigh(C)
    D = jnp.sqrt(jnp.where(D2 < 0, 1e-20, D2))
    C = jnp.dot(jnp.dot(B, jnp.diag(D ** 2)), B.T)
    return C, B, D


if __name__ == "__main__":
    from evosax.problems import batch_rosenbrock
    rng = jax.random.PRNGKey(0)
    strategy = CMA_ES(popsize=20, num_dims=2, elite_ratio=0.5)
    params = strategy.default_params
    state = strategy.initialize(rng, params)
    fitness_log = []
    num_iters = 25
    for t in range(num_iters):
        rng, rng_iter = jax.random.split(rng)
        y, state = strategy.ask(rng_iter, state, params)
        fitness = batch_rosenbrock(y, 1, 100)
        state = strategy.tell(y, fitness, state, params)
        best_id = jnp.argmin(fitness)
        print(t, fitness[best_id], state["mean"])
        fitness_log.append(fitness[best_id])


# def check_initialization(params):
#     """ Check lrates and other params of CMA-ES at initialization. """
#     assert population_size > 0, "popsize must be non-zero positive value."
#     assert n_dim > 1, "The dimension of mean must be larger than 1"
#     assert sigma > 0, "sigma must be non-zero positive value"
#     assert c_1 <= 1 - c_mu, "invalid lrate for the rank-one update"
#     assert c_mu <= 1 - c_1, "invalid lrate for the rank-μ update"
#     assert c_sigma < 1, "invalid lrate for cum. of step-size c."
#     assert c_c <= 1, "invalid lrate for cum. of rank-one update"
#     return
#
#
# def check_termination(values, params, state):
#     """ Check whether to terminate CMA-ES loop. """
#     dC = jnp.diag(memory["C"])
#     C, B, D = eigen_decomposition(memory["C"], memory["B"], memory["D"])
#
#     # Stop if generation fct values of recent generation is below thresh.
#     if (memory["generation"] > params["min_generations"]
#         and jnp.max(values) - jnp.min(values) < params["tol_fun"]):
#         print("TERMINATE ----> Convergence/No progress in objective")
#         return True
#
#     # Stop if std of normal distrib is smaller than tolx in all coordinates
#     # and pc is smaller than tolx in all components.
#     if jnp.all(memory["sigma"] * dC < params["tol_x"]) and np.all(
#         memory["sigma"] * memory["p_c"] < params["tol_x"]):
#         print("TERMINATE ----> Convergence/Search variance too small")
#         return True
#
#     # Stop if detecting divergent behavior.
#     if memory["sigma"] * jnp.max(D) > params["tol_x_up"]:
#         print("TERMINATE ----> Stepsize sigma exploded")
#         return True
#
#     # No effect coordinates: stop if adding 0.2-standard deviations
#     # in any single coordinate does not change m.
#     if jnp.any(memory["mean"] == memory["mean"] + (0.2 * memory["sigma"] * jnp.sqrt(dC))):
#         print("TERMINATE ----> No effect when adding std to mean")
#         return True
#
#     # No effect axis: stop if adding 0.1-standard deviation vector in
#     # any principal axis direction of C does not change m.
#     if jnp.all(memory["mean"] == memory["mean"] + (0.1 * memory["sigma"]
#                                 * D[0] * B[:, 0])):
#         print("TERMINATE ----> No effect when adding std to mean")
#         return True
#
#     # Stop if the condition number of the covariance matrix exceeds 1e14.
#     condition_cov = jnp.max(D) / jnp.min(D)
#     if condition_cov > params["tol_condition_C"]:
#         print("TERMINATE ----> C condition number exploded")
#         return True
#     return False
