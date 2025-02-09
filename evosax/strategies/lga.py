import pkgutil

import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..learned_eo.les_tools import (
    FitnessFeatures,
    load_pkl_object,
)
from ..learned_eo.lga_tools import (
    MutationAttention,
    SamplingAttention,
    SelectionAttention,
    tanh_age,
)
from ..strategy import Strategy


@struct.dataclass
class State:
    key: jax.Array
    mean: chex.Array
    archive_age: chex.Array  # Parents: 'Age' counter
    archive_x: chex.Array  # Parents: Solution vectors
    archive_f: chex.Array  # Parents: Fitness scores
    archive_sigma: chex.Array  # Parents: Mutation strengths
    sigma_C: chex.Array  # Children: Mutation strengths
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class Params:
    net_params: chex.ArrayTree
    cross_over_rate: float = 0.0
    sigma_init: float = 1.0
    init_min: float = -5.0
    init_max: float = 5.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class LGA(Strategy):
    """Population-Invariant Learned Genetic Algorithm (Lange et al., 2023)."""

    # NOTE: This is an independent reimplementation which does not use the same
    # meta-trained checkpoint used to generate the results in the paper. It
    # has been independently meta-trained & tested on a handful of Brax tasks.
    # The results may therefore differ from the ones shown in the paper.
    def __init__(
        self,
        population_size: int,
        solution: chex.ArrayTree | chex.Array | None = None,
        elite_ratio: float = 1.0,
        net_params: chex.ArrayTree | None = None,
        net_ckpt_path: str | None = None,
        sigma_init: float = 1.0,
        **fitness_kwargs: bool | int | float,
    ):
        super().__init__(
            population_size,
            solution,
            **fitness_kwargs,
        )
        self.strategy_name = "LGA"
        self.elite_ratio = elite_ratio
        self.elite_population_size = max(
            1, int(self.population_size * self.elite_ratio)
        )
        self.fitness_features = FitnessFeatures(centered_rank=True, z_score=True)
        self.sigma_init = sigma_init
        self.selection_layer = SelectionAttention(2, 16)
        self.sampling_layer = SamplingAttention(2, 16)
        self.mutation_layer = MutationAttention(2, 16)

        # Set net params provided at instantiation
        if net_params is not None:
            self.lga_net_params = net_params

        # Load network weights from checkpoint
        if net_ckpt_path is not None:
            self.lga_net_params = load_pkl_object(net_ckpt_path)
            print(f"Loaded LGA model from ckpt: {net_ckpt_path}")

        if net_params is None and net_ckpt_path is None:
            if self.num_dims > 50:
                ckpt_fname = "2023_04_lga_v7.pkl"
            else:
                ckpt_fname = "2023_04_lga_v4.pkl"
            data = pkgutil.get_data(__name__, f"ckpt/lga/{ckpt_fname}")
            self.lga_net_params = load_pkl_object(data, pkg_load=True)
            print(f"Loaded pretrained LGA model from ckpt: {ckpt_fname}")

    @property
    def params_strategy(self) -> Params:
        """Return default parameters of evolution strategy."""
        return Params(net_params=self.lga_net_params, sigma_init=self.sigma_init)

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """`init` the evolution strategy."""
        init_x = jax.random.uniform(
            key,
            (self.elite_population_size, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        init_sigma = jnp.ones((self.elite_population_size, 1)) * params.sigma_init

        return State(
            key=key,
            mean=init_x[0],
            archive_x=init_x,
            archive_f=jnp.zeros(self.elite_population_size) + 5000000.0,
            archive_sigma=init_sigma,
            archive_age=jnp.zeros(self.elite_population_size),
            sigma_C=jnp.zeros((self.population_size, 1)),
            best_member=init_x[0],
        )

    def ask_strategy(
        self, key_epsilon: jax.Array, state: State, params: Params
    ) -> tuple[chex.Array, State]:
        """`ask` for new parameter candidates to evaluate next."""
        key_idx, key_epsilon = jax.random.split(key_epsilon)

        elite_ids = jnp.arange(self.elite_population_size)

        # Sample candidates with replacement given distribution
        # Get probabilities to sample children from parent archive
        age_features = tanh_age(state.archive_age, state.generation_counter + 1e-10)
        F_E = self.fitness_features.apply(
            state.archive_x, state.archive_f, state.best_fitness
        )
        F_E = jnp.concatenate([F_E, age_features.reshape(-1, 1)], axis=1)
        p = self.sampling_layer.apply(params.net_params["sampling"], F_E)
        idx = jax.random.choice(key_idx, elite_ids, (self.population_size,), p=p)

        # Select children with sampled indices
        X_C = state.archive_x[idx]
        f_C = state.archive_f[idx]
        sigma_C = state.archive_sigma[idx]

        # Perform mutation rate adaptation of solutions
        F_C_tilde = self.fitness_features.apply(X_C, f_C, state.best_fitness)
        sigma_C = self.mutation_layer.apply(
            params.net_params["mutation"], sigma_C, F_C_tilde
        )

        # Perform Gaussian scaled mutation
        epsilon = jax.random.normal(key_epsilon, (self.population_size, self.num_dims))
        x = X_C + sigma_C * epsilon
        return x, state.replace(sigma_C=sigma_C)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: State,
        params: Params,
    ) -> State:
        """`tell` performance data for strategy state update."""
        fit_all = jnp.concatenate([fitness, state.archive_f])
        x_all = jnp.concatenate([x, state.archive_x])
        fit_re = self.fitness_features.apply(x_all, fit_all, state.best_fitness)
        idx = jnp.argsort(fit_all)[: self.elite_population_size]

        key, key_selection = jax.random.split(state.key)
        # Perform selection - either learned or hard truncation based
        F_X, F_E = fit_re[: self.population_size], fit_re[self.population_size :]
        select_bool = self.selection_layer.apply(
            params.net_params["selection"], key_selection, F_X, F_E
        )
        keep_parent = (select_bool.sum(axis=1) == 0)[:, None]
        next_x = select_bool @ x + keep_parent * state.archive_x
        next_f = select_bool @ fitness + keep_parent.squeeze() * state.archive_f
        next_sigma = select_bool @ state.sigma_C + keep_parent * state.archive_sigma

        # Update the age counter - reset if copy over otherwise increase
        next_age = state.archive_age * keep_parent.squeeze() + keep_parent.squeeze()

        # Argsort by performance and set mean
        improved = fit_all[idx][0] < state.best_fitness
        best_mean = jax.lax.select(improved, x_all[idx][0], state.best_member)
        return state.replace(
            key=key,
            mean=best_mean,
            archive_x=next_x,
            archive_f=next_f,
            archive_sigma=next_sigma,
            archive_age=next_age,
        )
