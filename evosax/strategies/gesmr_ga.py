from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..strategy import Strategy


@struct.dataclass
class EvoState:
    rng: chex.PRNGKey
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    sigma: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


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
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        elite_ratio: float = 0.5,
        sigma_ratio: float = 0.5,
        sigma_init: float = 0.07,
        sigma_meta: float = 2.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Group Elite Selection of Mutation Rates (GESMR) GA."""

        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            n_devices=n_devices,
            **fitness_kwargs
        )
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
        self.num_sigma_groups = int(jnp.sqrt(self.popsize))
        self.members_per_group = int(
            jnp.ceil(self.popsize / self.num_sigma_groups)
        )
        self.sigma_ratio = sigma_ratio
        self.sigma_popsize = max(
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

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the differential evolution strategy."""
        rng, rng_init = jax.random.split(rng)
        initialization = jax.random.uniform(
            rng_init,
            (self.elite_popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            rng=rng,
            mean=initialization[0],
            archive=initialization,
            fitness=jnp.zeros(self.elite_popsize) + jnp.finfo(jnp.float32).max,
            sigma=jnp.zeros(self.num_sigma_groups) + params.sigma_init,
            best_member=initialization[0],
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        rng, rng_idx, rng_eps_x, rng_eps_s = jax.random.split(rng, 4)
        # Sample noise for mutation of x and sigma
        eps_x = jax.random.normal(rng_eps_x, (self.popsize, self.num_dims))
        eps_s = jax.random.uniform(
            rng_eps_s, (self.num_sigma_groups,), minval=-1, maxval=1
        )

        # Sample members to evaluate from parent archive
        idx = jax.random.choice(
            rng_idx, jnp.arange(self.elite_popsize), (self.popsize - 1,)
        )
        x = jnp.concatenate([state.archive[0][None, :], state.archive[idx]])

        # Store fitness before perturbation (used to compute meta-fitness)
        fitness_mem = jnp.concatenate(
            [state.fitness[0][None], state.fitness[idx]]
        )

        # Apply sigma mutation on group level -> repeat for popmember broadcast
        sigma_perturb = state.sigma * params.sigma_meta ** eps_s
        sigma_repeated = jnp.repeat(sigma_perturb, self.members_per_group)[
            : self.popsize
        ]
        sigma = jnp.concatenate([state.sigma[0][None], sigma_repeated[1:]])

        # Apply x mutation -> scale specific to group membership
        x += sigma[:, None] * eps_x
        return x, state.replace(
            archive=x, fitness=fitness_mem, sigma=sigma_perturb
        )

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        # Select best x members
        idx = jnp.argsort(fitness)[: self.elite_popsize]
        archive = x[idx]

        # Select best sigma based on function value improvement
        group_ids = jnp.repeat(
            jnp.arange(self.members_per_group), self.num_sigma_groups
        )[: self.popsize]
        delta_fitness = fitness - state.fitness

        best_deltas = []
        for k in range(self.num_sigma_groups):
            sub_mask = group_ids == k
            sub_delta = (
                sub_mask * delta_fitness
                + (1 - sub_mask) * jnp.finfo(jnp.float32).max
            )
            max_sub_delta = jnp.min(sub_delta)
            best_deltas.append(max_sub_delta)

        idx_select = jnp.argsort(jnp.array(best_deltas))[: self.sigma_popsize]
        sigma_elite = state.sigma[idx_select]

        # Resample sigmas with replacement
        rng, rng_sigma = jax.random.split(state.rng)
        idx_s = jax.random.choice(
            rng_sigma,
            jnp.arange(self.sigma_popsize),
            (self.num_sigma_groups - 1,),
        )
        sigma = jnp.concatenate([state.sigma[0][None], sigma_elite[idx_s]])

        # Set mean to best member seen so far
        improved = fitness[0] < state.best_fitness
        best_mean = jax.lax.select(improved, archive[0], state.best_member)
        return state.replace(
            rng=rng,
            fitness=fitness[idx],
            archive=archive,
            sigma=sigma,
            mean=best_mean,
        )
