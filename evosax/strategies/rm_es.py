import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class RmES(Strategy):
    def __init__(
        self,
        num_dims: int,
        popsize: int,
        elite_ratio: float = 0.5,
        memory_size: int = 10,
    ):
        """Rank-m ES (Li & Zhang, 2017)
        Reference: https://ieeexplore.ieee.org/document/8080257
        """
        super().__init__(num_dims, popsize)
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.memory_size = memory_size  # number of ranks
        self.strategy_name = "RmES"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        weights = jnp.array(
            [
                (
                    (jnp.log(self.elite_popsize + 1) - jnp.log(i + 1))
                    / (
                        self.elite_popsize * jnp.log(self.elite_popsize + 1)
                        - jnp.sum(
                            jnp.log(jnp.arange(1, self.elite_popsize + 1))
                        )
                    )
                )
                for i in range(self.elite_popsize)
            ]
        )
        mu_eff = 1 / jnp.sum(weights ** 2)
        c_cov = 1 / (3 * jnp.sqrt(self.num_dims) + 5)
        c_c = 2 / (self.num_dims + 7)
        params = {
            "weights": weights,
            "c_cov": c_cov,
            "c_c": c_c,
            "c_m": 1.0,
            "c_sigma": jnp.minimum(2 / (self.num_dims + 7), 0.05),
            "mu_eff": mu_eff,
            "sigma_init": 1.0,
            "sigma_limit": 0.001,
            "t_uncorr": 20,
            "q_star": 0.325,
            "c_s": 0.3,
            "d_sigma": 1.0,
            "init_min": 0.0,
            "init_max": 0.0,
        }
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        # Initialize evolution paths & covariance matrix
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "p_sigma": jnp.zeros(self.num_dims),
            "sigma": params["sigma_init"],
            "mean": initialization,
            "P": jnp.zeros((self.num_dims, self.memory_size)),
            "t_gap": jnp.zeros(self.memory_size),
            "s_rank_rate": 0,
            # Store previous generations fitness for rank-based success rule
            "fitness_archive": jnp.zeros(self.popsize) + 1e20,
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        x = sample(
            rng,
            state["mean"],
            state["sigma"],
            state["P"],
            self.num_dims,
            self.popsize,
            params["c_cov"],
            state["gen_counter"],
        )
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()][
            : self.elite_popsize
        ]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean = update_mean(state["mean"], sorted_solutions, params)
        p_sigma = update_p_sigma(
            state["mean"],
            mean,
            state["sigma"],
            state["p_sigma"],
            params,
        )
        P, t_gap = update_P_matrix(
            state["P"],
            state["p_sigma"],
            state["t_gap"],
            params["t_uncorr"],
            state["gen_counter"],
        )

        s_rank_rate = rank_success_rule(
            fitness,
            state["fitness_archive"],
            state["s_rank_rate"],
            params["q_star"],
            params["weights"],
            params["c_s"],
        )
        sigma = update_sigma(state["sigma"], s_rank_rate, params)
        sigma = jnp.maximum(sigma, params["sigma_limit"])
        state["mean"] = mean
        state["p_sigma"] = p_sigma
        state["P"] = P
        state["t_gap"] = t_gap
        state["sigma"] = sigma
        state["s_rank_rate"] = s_rank_rate
        state["fitness_archive"] = fitness
        return state


def update_mean(
    mean: chex.Array,
    sorted_solutions: chex.Array,
    params: chex.ArrayTree,
) -> chex.Array:
    """Update mean of strategy."""
    mean = (1 - params["c_m"]) * mean + params["c_m"] * jnp.sum(
        sorted_solutions[:, 1:].T * params["weights"], axis=1
    )
    return mean


def update_p_sigma(
    mean_old: chex.Array,
    mean: chex.Array,
    sigma: float,
    p_sigma: chex.Array,
    params: chex.ArrayTree,
) -> chex.Array:
    """Update evolution path for covariance matrix."""
    p_sigma_new = (1 - params["c_sigma"]) * p_sigma + jnp.sqrt(
        params["c_sigma"] * (2 - params["c_sigma"]) * params["mu_eff"]
    ) * (mean - mean_old) / sigma
    return p_sigma_new


def update_P_matrix(
    P: chex.Array,
    p_sigma: chex.Array,
    t_gap: chex.Array,
    t_uncorr: int,
    gen_counter: int,
) -> Tuple[chex.Array, chex.Array]:
    """Update the P matrix storing m evolution paths."""
    memory_size = P.shape[1]
    # Use evo paths in separated generations - keep them uncorrelated!
    T_min = jnp.min(t_gap[1:] - t_gap[:-1])
    replace_crit = T_min > t_uncorr
    fill_up_crit = gen_counter < memory_size
    push_replace = jnp.logical_or(replace_crit, fill_up_crit)

    # Case 1: Initially Record all evolution paths - make space for new one
    P_c1 = P.at[:, :-1].set(P[:, 1:])
    t_gap_c1 = t_gap.at[:-1].set(t_gap[1:])

    # Case 2: Remove/overwrite the oldesy recorded evolution path
    # Problem: i_min is a dynamic index - write as sum of two matrices?!
    P_c2 = P[:]
    t_gap_c2 = t_gap[:]
    i_min = jnp.argmin(t_gap[:-1] - t_gap[1:])
    for i in range(memory_size - 1):
        replace_bool = i >= i_min
        P_c2 = jax.lax.select(
            replace_bool, P_c2.at[:, i].set(P_c2[:, i + 1]), P_c2
        )
        t_gap_c2 = jax.lax.select(
            replace_bool, t_gap_c2.at[i].set(t_gap_c2[i + 1]), t_gap_c2
        )

    P = jax.lax.select(push_replace, P_c1, P_c2)
    t_gap = jax.lax.select(push_replace, t_gap_c1, t_gap_c2)

    # Finally update with the most recent evolution path
    P = P.at[:, memory_size - 1].set(p_sigma)
    t_gap = t_gap.at[memory_size - 1].set(gen_counter)
    return P, t_gap


def update_sigma(
    sigma: float, s_rank_rate: float, params: chex.ArrayTree
) -> float:
    """Update stepsize sigma."""
    sigma_new = sigma * jnp.exp(s_rank_rate / params["d_sigma"])
    return sigma_new


def sample(
    rng: chex.PRNGKey,
    mean: chex.Array,
    sigma: float,
    P: chex.Array,
    n_dim: int,
    pop_size: int,
    c_cov: chex.Array,
    gen_counter: int,
) -> chex.Array:
    """Jittable Gaussian Sample Helper."""
    z = jax.random.normal(rng, (n_dim, pop_size))  # ~ N(0, I)
    r = jax.random.normal(rng, (n_dim, P.shape[1]))

    for j in range(P.shape[1]):
        update_bool = gen_counter > j
        new_z = (
            jnp.sqrt(1 - c_cov) * z
            + (jnp.sqrt(c_cov) * P[:, j])[:, jnp.newaxis]
            * r[:, j][:, jnp.newaxis]
        )
        z = jax.lax.select(update_bool, new_z, z)
    z = jnp.swapaxes(z, 1, 0)
    x = mean + sigma * z  # ~ N(m, σ^2 C)
    return x


def rank_success_rule(
    fitness: chex.Array,
    fitness_archive: chex.Array,
    s_rank_rate: float,
    q_star: float,
    weights: chex.Array,
    c_s: float,
) -> float:
    """Compute rank-based success rule (cumulative rank rate)."""
    elite_popsize = weights.shape[0]
    popsize = fitness.shape[0]

    # Step 1: Sort all fitnesses in ascending order and get ranks
    # Rank parents + kids jointly - subdivide afterwards & take elite from both
    concat_all = jnp.vstack(
        [jnp.expand_dims(fitness, 1), jnp.expand_dims(fitness_archive, 1)]
    )
    ranks = jnp.zeros(concat_all.shape[0])
    ranks = ranks.at[concat_all[:, 0].argsort()].set(jnp.arange(2 * popsize))

    ranks_current = ranks[:popsize]
    ranks_current = ranks_current[ranks_current.argsort()][:elite_popsize]
    ranks_last = ranks[popsize:]
    ranks_last = ranks_last[ranks_last.argsort()][:elite_popsize]
    ranks_current, ranks_last

    # Step 2: Compute rank difference (Parents vs. kids) - paper assumes min!
    q = 1 / elite_popsize * jnp.sum(weights * (ranks_last - ranks_current))

    # Step 3: Compute comulative rank rate using decaying memory
    new_s_rank_rate = (1 - c_s) * s_rank_rate + c_s * (q - q_star)
    return new_s_rank_rate
