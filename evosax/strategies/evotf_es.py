"""Evolution Transformer (Lange et al., 2024).

Reference: https://arxiv.org/abs/2403.02985
"""

import pkgutil
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..learned_eo.evotf_tools import (
    DistributionFeaturizer,
    EvoTransformer,
    FitnessFeaturizer,
    SolutionFeaturizer,
)
from ..learned_eo.les_tools import load_pkl_object
from ..strategy import Params, State, Strategy, metrics_fn
from ..types import Fitness, Population, PyTree, Solution


@struct.dataclass
class DistributionFeaturesState:
    old_mean: jax.Array
    old_std: jax.Array
    momentum_mean: jax.Array
    momentum_std: jax.Array
    evopath_mean: jax.Array
    evopath_std: jax.Array


@struct.dataclass
class FitnessFeaturesState:
    best_fitness: float


@struct.dataclass
class SolutionFeaturesState:
    best_fitness: float
    best_member: jax.Array
    generation_counter: int


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    sf_state: SolutionFeaturesState
    ff_state: FitnessFeaturesState
    df_state: DistributionFeaturesState
    solution_context: jax.Array
    fitness_context: jax.Array
    distribution_context: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    lrate_mean: float
    lrate_std: float
    params: PyTree


class EvoTF_ES(Strategy):
    """Evolution Transformer (EvoTF)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        max_context_len: int = 100,
        model_config: dict = dict(
            num_layers=1,
            num_heads=1,
            num_latents=8,
            latent_dim=32,
            embed_dim=128,
            use_fitness_encoder=True,
            use_dist_encoder=True,
            use_crossd_encoder=False,
        ),
        solution_config: dict = dict(
            norm_diff_mean=True,
            norm_diff_mean_sq=True,
            diff_best=True,
            norm_diff_best=True,
            maximize=False,
        ),
        fitness_config: dict = dict(
            improved_best=True,
            z_score=True,
            norm_diff_best=True,
            norm_range=True,
            snes_weights=True,
            des_weights=True,
            w_decay=0.00,
            maximize=False,
        ),
        distrib_config: dict = dict(
            use_mean=False,
            use_sigma=False,
            use_evopaths=True,
            use_momentum=True,
            evopath_timescales=[0.9, 0.95, 0.99],
            momentum_timescales=[0.9, 0.95, 0.99],
            use_oai_grad=True,
        ),
        use_antithetic_sampling: bool = False,
        params: PyTree | None = None,
        params_path: str | None = None,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "EvoTransformer"

        self.max_context_len = max_context_len
        self.model_config = model_config
        self.solution_config = solution_config
        self.fitness_config = fitness_config
        self.distrib_config = distrib_config
        self.model = EvoTransformer(**model_config)
        self.sf = SolutionFeaturizer(
            population_size=self.population_size,
            num_dims=self.num_dims,
            seq_len=self.max_context_len,
            **solution_config,
        )
        self.ff = FitnessFeaturizer(
            population_size=self.population_size,
            num_dims=self.num_dims,
            seq_len=self.max_context_len,
            **fitness_config,
        )
        self.df = DistributionFeaturizer(
            population_size=self.population_size,
            num_dims=self.num_dims,
            seq_len=self.max_context_len,
            **distrib_config,
        )

        if params is not None:
            # Set params provided
            self.params = params
        elif params_path is not None:
            # Load params from checkpoint
            self.ckpt = load_pkl_object(params_path)
            self.params = self.ckpt["net_params"]
            self.model_config = self.ckpt["model_config"]
            self.model = EvoTransformer(**self.model_config)
            self.sf = SolutionFeaturizer(
                population_size=self.population_size,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["solution_config"],
            )
            self.ff = FitnessFeaturizer(
                population_size=self.population_size,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["fitness_config"],
            )
            self.df = DistributionFeaturizer(
                population_size=self.population_size,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["distrib_config"],
            )
        else:
            # Load default params
            ckpt_fname = "2024_03_SNES_small.pkl"
            data = pkgutil.get_data(__name__, f"ckpt/evotf/{ckpt_fname}")
            self.ckpt = load_pkl_object(data, pkg_load=True)
            self.params = self.ckpt["net_params"]
            self.model_config = self.ckpt["model_config"]
            self.model = EvoTransformer(**self.model_config)
            self.sf = SolutionFeaturizer(
                population_size=self.population_size,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["solution_config"],
            )
            self.ff = FitnessFeaturizer(
                population_size=self.population_size,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["fitness_config"],
            )
            self.df = DistributionFeaturizer(
                population_size=self.population_size,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["distrib_config"],
            )

        self.num_network_params = sum(x.size for x in jax.tree.leaves(self.params))

        # Antithetic sampling
        self.use_antithetic_sampling = use_antithetic_sampling
        if self.use_antithetic_sampling:
            assert self.population_size % 2 == 0, "Population size must be even."

        # Setup mask for forward pass
        self.la_mask = jnp.tril(jnp.ones((self.max_context_len, self.max_context_len)))

    @property
    def _default_params(self) -> Params:
        params = Params(
            std_init=1.0,
            lrate_mean=1.0,
            lrate_std=1.0,
            params=self.params,
        )
        return params

    def _init(self, key: jax.Array, params: Params) -> State:
        scon_shape = (
            1,
            self.max_context_len,
            self.population_size,
            self.num_dims,
            self.sf.num_features,
        )
        fcon_shape = (
            1,
            self.max_context_len,
            self.population_size,
            self.ff.num_features,
        )
        dcon_shape = (
            1,
            self.max_context_len,
            self.num_dims,
            self.df.num_features,
        )
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
            sf_state=self.sf.init(),
            ff_state=self.ff.init(),
            df_state=self.df.init(),
            solution_context=jnp.zeros(scon_shape),
            fitness_context=jnp.zeros(fcon_shape),
            distribution_context=jnp.zeros(dcon_shape),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        if not self.use_antithetic_sampling:
            z = jax.random.normal(key, (self.population_size, self.num_dims))
        else:
            z = jax.random.normal(key, (self.population_size // 2, self.num_dims))
            z = jnp.concatenate([z, -z])

        population = state.mean + z * state.std
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Get features from population info
        sfeatures, sf_state = self.sf.featurize(
            population, fitness, state.mean, state.std, state.sf_state
        )
        ffeatures, ff_state = self.ff.featurize(population, fitness, state.ff_state)
        dfeatures, df_state = self.df.featurize(
            population, fitness, state.mean, state.std, state.df_state
        )

        # Update the context with a sliding window
        shift_buffer = state.generation_counter >= self.max_context_len
        idx = jax.lax.select(
            shift_buffer, self.max_context_len - 1, state.generation_counter
        )
        solution_context = jax.lax.select(
            shift_buffer,
            state.solution_context.at[0, :-1].set(state.solution_context[0, 1:]),
            state.solution_context,
        )
        fitness_context = jax.lax.select(
            shift_buffer,
            state.fitness_context.at[0, :-1].set(state.fitness_context[0, 1:]),
            state.fitness_context,
        )
        distribution_context = jax.lax.select(
            shift_buffer,
            state.distribution_context.at[0, :-1].set(
                state.distribution_context[0, 1:]
            ),
            state.distribution_context,
        )
        solution_context = solution_context.at[0, idx].set(sfeatures)
        fitness_context = fitness_context.at[0, idx].set(ffeatures)
        distribution_context = distribution_context.at[0, idx].set(dfeatures)

        # Update strategy via evotf forward pass
        @jax.jit
        def infer_step(solution_context, fitness_context, distribution_context):
            return self.model.apply(
                params.params,
                solution_context,
                fitness_context,
                distribution_context,
                rngs={"dropout": jax.random.key(0)},
                train=False,
                mask=self.la_mask,
            )

        pred, _ = infer_step(
            solution_context, fitness_context, distribution_context
        )  # TODO: att not used?
        pred_mean = state.mean + params.lrate_mean * state.std * pred[0, 0, idx]
        pred_std = state.std * jnp.exp(params.lrate_std / 2 * pred[1, 0, idx])
        pred_std = jnp.clip(pred_std, 1e-8)

        # Collect and sort attention by population fitness order
        return state.replace(
            mean=pred_mean,
            std=pred_std,
            sf_state=sf_state,
            ff_state=ff_state,
            df_state=df_state,
            solution_context=solution_context,
            fitness_context=fitness_context,
            distribution_context=distribution_context,
        )
