import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy
from ..types import Fitness, Population, Solution


@struct.dataclass
class State:
    p_sigma: jax.Array
    mean: jax.Array
    sigma: float
    P: jax.Array
    t_gap: jax.Array
    s_rank_rate: float
    fitness_archive: jax.Array
    weights: jax.Array
    best_member: jax.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class Params:
    c_cov: float
    c_c: float
    c_sigma: float
    mu_eff: float
    c_m: float = 1.0
    sigma_init: float = 1.0
    sigma_limit: float = 0.001
    t_uncorr: int = 20
    q_star: float = 0.325
    c_s: float = 0.3
    d_sigma: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


def get_elite_weights(elite_population_size: int) -> tuple[jax.Array, jax.Array]:
    """Utility helper to create truncated elite weights for mean update."""
    weights = jnp.array(
        [
            (
                (jnp.log(elite_population_size + 1) - jnp.log(i + 1))
                / (
                    elite_population_size * jnp.log(elite_population_size + 1)
                    - jnp.sum(jnp.log(jnp.arange(1, elite_population_size + 1)))
                )
            )
            for i in range(elite_population_size)
        ]
    )
    return weights


class RmES(Strategy):
    def __init__(
        self,
        population_size: int,
        solution: Solution,
        elite_ratio: float = 0.5,
        memory_size: int = 10,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Rank-m ES (Li & Zhang, 2017)
        Reference: https://ieeexplore.ieee.org/document/8080257
        """
        super().__init__(population_size, solution, mean_decay, **fitness_kwargs)
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_population_size = max(
            1, int(self.population_size * self.elite_ratio)
        )
        self.memory_size = memory_size  # number of ranks
        self.strategy_name = "RmES"

        # Set core kwargs params
        self.sigma_init = sigma_init

    @property
    def params_strategy(self) -> Params:
        """Return default parameters of evolution strategy."""
        weights = get_elite_weights(self.elite_population_size)
        mu_eff = 1 / jnp.sum(weights**2)
        c_cov = 1 / (3 * jnp.sqrt(self.num_dims) + 5)
        c_c = 2 / (self.num_dims + 7)
        params = Params(
            c_cov=c_cov,
            c_c=c_c,
            c_sigma=jnp.minimum(2 / (self.num_dims + 7), 0.05),
            mu_eff=mu_eff,
            sigma_init=self.sigma_init,
        )
        return params

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """`init` the evolution strategy."""
        weights = get_elite_weights(self.elite_population_size)
        # Initialize evolution paths & covariance matrix
        initialization = jax.random.uniform(
            key,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = State(
            p_sigma=jnp.zeros(self.num_dims),
            sigma=params.sigma_init,
            mean=initialization,
            P=jnp.zeros((self.num_dims, self.memory_size)),
            t_gap=jnp.zeros(self.memory_size),
            s_rank_rate=0.0,
            weights=weights,
            # Store previous generations fitness for rank-based success rule
            fitness_archive=jnp.zeros(self.population_size) + 1e20,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[jax.Array, State]:
        """`ask` for new parameter candidates to evaluate next."""
        x = sample(
            key,
            state.mean,
            state.sigma,
            state.P,
            self.num_dims,
            self.population_size,
            params.c_cov,
            state.generation_counter,
        )
        return x, state

    def tell_strategy(
        self,
        x: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        """`tell` performance data for strategy state update."""
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()][
            : self.elite_population_size
        ]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean = update_mean(state.mean, sorted_solutions, params.c_m, state.weights)
        p_sigma = update_p_sigma(
            state.mean,
            mean,
            state.sigma,
            state.p_sigma,
            params.c_sigma,
            params.mu_eff,
        )
        P, t_gap = update_P_matrix(
            state.P,
            state.p_sigma,
            state.t_gap,
            params.t_uncorr,
            state.generation_counter,
        )

        s_rank_rate = rank_success_rule(
            fitness,
            state.fitness_archive,
            state.s_rank_rate,
            params.q_star,
            state.weights,
            params.c_s,
        )
        sigma = update_sigma(state.sigma, s_rank_rate, params.d_sigma)
        sigma = jnp.maximum(sigma, params.sigma_limit)
        return state.replace(
            mean=mean,
            p_sigma=p_sigma,
            P=P,
            t_gap=t_gap,
            sigma=sigma,
            s_rank_rate=s_rank_rate,
            fitness_archive=fitness,
        )


def update_mean(
    mean: jax.Array,
    sorted_solutions: jax.Array,
    c_m: float,
    weights: jax.Array,
) -> jax.Array:
    """Update mean of strategy."""
    mean = (1 - c_m) * mean + c_m * jnp.sum(sorted_solutions[:, 1:].T * weights, axis=1)
    return mean


def update_p_sigma(
    mean_old: jax.Array,
    mean: jax.Array,
    sigma: float,
    p_sigma: jax.Array,
    c_sigma: float,
    mu_eff: float,
) -> jax.Array:
    """Update evolution path for covariance matrix."""
    p_sigma_new = (1 - c_sigma) * p_sigma + jnp.sqrt(
        c_sigma * (2 - c_sigma) * mu_eff
    ) * (mean - mean_old) / sigma
    return p_sigma_new


def update_P_matrix(
    P: jax.Array,
    p_sigma: jax.Array,
    t_gap: jax.Array,
    t_uncorr: int,
    generation_counter: int,
) -> tuple[jax.Array, jax.Array]:
    """Update the P matrix storing m evolution paths."""
    memory_size = P.shape[1]
    # Use evo paths in separated generations - keep them uncorrelated!
    T_min = jnp.min(t_gap[1:] - t_gap[:-1])
    replace_crit = T_min > t_uncorr
    fill_up_crit = generation_counter < memory_size
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
        P_c2 = jax.lax.select(replace_bool, P_c2.at[:, i].set(P_c2[:, i + 1]), P_c2)
        t_gap_c2 = jax.lax.select(
            replace_bool, t_gap_c2.at[i].set(t_gap_c2[i + 1]), t_gap_c2
        )

    P = jax.lax.select(push_replace, P_c1, P_c2)
    t_gap = jax.lax.select(push_replace, t_gap_c1, t_gap_c2)

    # Finally update with the most recent evolution path
    P = P.at[:, memory_size - 1].set(p_sigma)
    t_gap = t_gap.at[memory_size - 1].set(generation_counter)
    return P, t_gap


def update_sigma(sigma: float, s_rank_rate: float, d_sigma: float) -> float:
    """Update stepsize sigma."""
    sigma_new = sigma * jnp.exp(s_rank_rate / d_sigma)
    return sigma_new


def sample(
    key: jax.Array,
    mean: jax.Array,
    sigma: float,
    P: jax.Array,
    n_dim: int,
    pop_size: int,
    c_cov: jax.Array,
    generation_counter: int,
) -> jax.Array:
    """Jittable Gaussian Sample Helper."""
    key_z, key_r = jax.random.split(key, 2)
    z = jax.random.normal(key_z, (n_dim, pop_size))  # ~ N(0, I)
    r = jax.random.normal(key_r, (n_dim, P.shape[1]))

    for j in range(P.shape[1]):
        update_bool = generation_counter > j
        new_z = (
            jnp.sqrt(1 - c_cov) * z
            + (jnp.sqrt(c_cov) * P[:, j])[:, jnp.newaxis] * r[:, j][:, jnp.newaxis]
        )
        z = jax.lax.select(update_bool, new_z, z)
    z = jnp.swapaxes(z, 1, 0)
    x = mean + sigma * z  # ~ N(m, σ^2 C)
    return x


def rank_success_rule(
    fitness: Fitness,
    fitness_archive: Fitness,
    s_rank_rate: float,
    q_star: float,
    weights: jax.Array,
    c_s: float,
) -> float:
    """Compute rank-based success rule (cumulative rank rate)."""
    elite_population_size = weights.shape[0]
    population_size = fitness.shape[0]

    # Step 1: Sort all fitnesses in ascending order and get ranks
    # Rank parents + kids jointly - subdivide afterwards & take elite from both
    concat_all = jnp.vstack(
        [jnp.expand_dims(fitness, 1), jnp.expand_dims(fitness_archive, 1)]
    )
    ranks = jnp.zeros(concat_all.shape[0])
    ranks = ranks.at[concat_all[:, 0].argsort()].set(jnp.arange(2 * population_size))

    ranks_current = ranks[:population_size]
    ranks_current = ranks_current[ranks_current.argsort()][:elite_population_size]
    ranks_last = ranks[population_size:]
    ranks_last = ranks_last[ranks_last.argsort()][:elite_population_size]

    # Step 2: Compute rank difference (Parents vs. kids) - paper assumes min!
    q = 1 / elite_population_size * jnp.sum(weights * (ranks_last - ranks_current))

    # Step 3: Compute comulative rank rate using decaying memory
    new_s_rank_rate = (1 - c_s) * s_rank_rate + c_s * (q - q_star)
    return new_s_rank_rate
