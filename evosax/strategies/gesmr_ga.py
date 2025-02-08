import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy


@struct.dataclass
class EvoState:
    key: jax.Array
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    sigma: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class EvoParams:
    sigma_init: float = 0.07
    sigma_meta: float = 2.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class GESMR_GA(Strategy):
    def __init__(
        self,
        population_size: int,
        pholder_params: chex.ArrayTree | chex.Array | None = None,
        elite_ratio: float = 0.5,
        sigma_ratio: float = 0.5,
        sigma_init: float = 0.07,
        sigma_meta: float = 2.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Group Elite Selection of Mutation Rates (GESMR) GA."""
        super().__init__(population_size, pholder_params, **fitness_kwargs)
        self.elite_ratio = elite_ratio
        self.elite_population_size = max(
            1, int(self.population_size * self.elite_ratio)
        )
        self.num_sigma_groups = int(jnp.sqrt(self.population_size))
        self.members_per_group = int(
            jnp.ceil(self.population_size / self.num_sigma_groups)
        )
        self.sigma_ratio = sigma_ratio
        self.sigma_population_size = max(
            1, int(self.num_sigma_groups * self.sigma_ratio)
        )
        self.strategy_name = "GESMR_GA"
        # Set core kwargs es_params
        self.sigma_init = sigma_init
        self.sigma_meta = sigma_meta

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(sigma_init=self.sigma_init, sigma_meta=self.sigma_meta)

    def init_strategy(self, key_state: jax.Array, params: EvoParams) -> EvoState:
        """`init` the differential evolution strategy."""
        key_init, key_state = jax.random.split(key_state)
        initialization = jax.random.uniform(
            key_init,
            (self.elite_population_size, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            key=key_state,
            mean=initialization[0],
            archive=initialization,
            fitness=jnp.zeros(self.elite_population_size) + jnp.finfo(jnp.float32).max,
            sigma=jnp.zeros(self.num_sigma_groups) + params.sigma_init,
            best_member=initialization[0],
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: EvoState, params: EvoParams
    ) -> tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        key_eps_x, key_eps_s, key_idx = jax.random.split(key, 3)
        # Sample noise for mutation of x and sigma
        eps_x = jax.random.normal(key_eps_x, (self.population_size, self.num_dims))
        eps_s = jax.random.uniform(
            key_eps_s, (self.num_sigma_groups,), minval=-1, maxval=1
        )

        # Sample members to evaluate from parent archive
        idx = jax.random.choice(
            key_idx, jnp.arange(self.elite_population_size), (self.population_size - 1,)
        )
        x = jnp.concatenate([state.archive[0][None, :], state.archive[idx]])

        # Store fitness before perturbation (used to compute meta-fitness)
        fitness_mem = jnp.concatenate([state.fitness[0][None], state.fitness[idx]])

        # Apply sigma mutation on group level -> repeat for popmember broadcast
        sigma_perturb = state.sigma * params.sigma_meta**eps_s
        sigma_repeated = jnp.repeat(sigma_perturb, self.members_per_group)[
            : self.population_size
        ]
        sigma = jnp.concatenate([state.sigma[0][None], sigma_repeated[1:]])

        # Apply x mutation -> scale specific to group membership
        x += sigma[:, None] * eps_x
        return x, state.replace(archive=x, fitness=fitness_mem, sigma=sigma_perturb)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        # Select best x members
        idx = jnp.argsort(fitness)[: self.elite_population_size]
        archive = x[idx]

        # Select best sigma based on function value improvement
        group_ids = jnp.repeat(
            jnp.arange(self.members_per_group), self.num_sigma_groups
        )[: self.population_size]
        delta_fitness = fitness - state.fitness

        best_deltas = []
        for k in range(self.num_sigma_groups):
            sub_mask = group_ids == k
            sub_delta = (
                sub_mask * delta_fitness + (1 - sub_mask) * jnp.finfo(jnp.float32).max
            )
            max_sub_delta = jnp.min(sub_delta)
            best_deltas.append(max_sub_delta)

        idx_select = jnp.argsort(jnp.array(best_deltas))[: self.sigma_population_size]
        sigma_elite = state.sigma[idx_select]

        # Resample sigmas with replacement
        key, key_sigma = jax.random.split(state.key)
        idx_s = jax.random.choice(
            key_sigma,
            jnp.arange(self.sigma_population_size),
            (self.num_sigma_groups - 1,),
        )
        sigma = jnp.concatenate([state.sigma[0][None], sigma_elite[idx_s]])

        # Set mean to best member seen so far
        improved = fitness[0] < state.best_fitness
        best_mean = jax.lax.select(improved, archive[0], state.best_member)
        return state.replace(
            key=key,
            fitness=fitness[idx],
            archive=archive,
            sigma=sigma,
            mean=best_mean,
        )
