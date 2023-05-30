from typing import Optional, Tuple, Union
import chex
from flax import struct
import jax
import jax.numpy as jnp
import pkgutil
from ..learned_eo.les_tools import (
    FitnessFeatures,
    load_pkl_object,
)
from ..learned_eo.lga_tools import (
    SelectionAttention,
    SamplingAttention,
    MutationAttention,
    tanh_age,
)
from ..strategy import Strategy


@struct.dataclass
class EvoState:
    rng: chex.PRNGKey
    mean: chex.Array
    archive_age: chex.Array  # Parents: 'Age' counter
    archive_x: chex.Array  # Parents: Solution vectors
    archive_f: chex.Array  # Parents: Fitness scores
    archive_sigma: chex.Array  # Parents: Mutation strengths
    sigma_C: chex.Array  # Children: Mutation strengths
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
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
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        elite_ratio: float = 1.0,
        net_params: Optional[chex.ArrayTree] = None,
        net_ckpt_path: Optional[str] = None,
        sigma_init: float = 1.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float],
    ):
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            n_devices=n_devices,
            **fitness_kwargs,
        )
        self.strategy_name = "LGA"
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
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
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(net_params=self.lga_net_params, sigma_init=self.sigma_init)

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        init_x = jax.random.uniform(
            rng,
            (self.elite_popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        init_sigma = jnp.ones((self.elite_popsize, 1)) * params.sigma_init

        return EvoState(
            rng=rng,
            mean=init_x[0],
            archive_x=init_x,
            archive_f=jnp.zeros(self.elite_popsize) + 5000000.0,
            archive_sigma=init_sigma,
            archive_age=jnp.zeros(self.elite_popsize),
            sigma_C=jnp.zeros((self.popsize, 1)),
            best_member=init_x[0],
        )

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        rng, rng_idx = jax.random.split(rng)
        elite_ids = jnp.arange(self.elite_popsize)

        # Sample candidates with replacement given distribution
        # Get probabilities to sample children from parent archive
        age_features = tanh_age(state.archive_age, state.gen_counter + 1e-10)
        F_E = self.fitness_features.apply(
            state.archive_x, state.archive_f, state.best_fitness
        )
        F_E = jnp.concatenate([F_E, age_features.reshape(-1, 1)], axis=1)
        p = self.sampling_layer.apply(params.net_params["sampling"], F_E)
        idx = jax.random.choice(rng_idx, elite_ids, (self.popsize,), p=p)

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
        epsilon = jax.random.normal(rng, (self.popsize, self.num_dims))
        x = X_C + sigma_C * epsilon
        return x, state.replace(sigma_C=sigma_C)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        fit_all = jnp.concatenate([fitness, state.archive_f])
        x_all = jnp.concatenate([x, state.archive_x])
        fit_re = self.fitness_features.apply(x_all, fit_all, state.best_fitness)
        idx = jnp.argsort(fit_all)[: self.elite_popsize]

        rng, rng_next = jax.random.split(state.rng)
        # Perform selection - either learned or hard truncation based
        F_X, F_E = fit_re[: self.popsize], fit_re[self.popsize :]
        select_bool = self.selection_layer.apply(
            params.net_params["selection"], rng, F_X, F_E
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
            rng=rng_next,
            mean=best_mean,
            archive_x=next_x,
            archive_f=next_f,
            archive_sigma=next_sigma,
            archive_age=next_age,
        )
