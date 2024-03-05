from typing import Tuple, Optional, Union
import chex
from flax import struct
import jax
import jax.numpy as jnp
import pkgutil
from ..strategy import Strategy
from ..learned_eo.les_tools import load_pkl_object
from ..learned_eo.evotf_tools import (
    EvoTransformer,
    SolutionFeaturizer,
    FitnessFeaturizer,
    DistributionFeaturizer,
)


@struct.dataclass
class DistributionFeaturesState:
    old_mean: chex.Array
    old_sigma: chex.Array
    momentum_mean: chex.Array
    momentum_sigma: chex.Array
    evopath_mean: chex.Array
    evopath_sigma: chex.Array


@struct.dataclass
class FitnessFeaturesState:
    best_fitness: float


@struct.dataclass
class SolutionFeaturesState:
    best_fitness: float
    best_member: chex.Array
    gen_counter: int


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: chex.Array
    sf_state: SolutionFeaturesState
    ff_state: FitnessFeaturesState
    df_state: DistributionFeaturesState
    solution_context: chex.Array
    fitness_context: chex.Array
    distribution_context: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    net_params: chex.ArrayTree
    lrate_mean: float = 1.0
    lrate_sigma: float = 1.0
    sigma_init: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class EvoTF_ES(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        sigma_init: float = 1.0,
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
        net_params: Optional[chex.ArrayTree] = None,
        net_ckpt_path: Optional[str] = None,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float],
    ):
        super().__init__(
            popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs
        )
        self.strategy_name = "EvoTransformer"
        self.max_context_len = max_context_len
        self.model_config = model_config
        self.solution_config = solution_config
        self.fitness_config = fitness_config
        self.distrib_config = distrib_config
        self.model = EvoTransformer(**model_config)
        self.sf = SolutionFeaturizer(
            popsize=self.popsize,
            num_dims=self.num_dims,
            seq_len=self.max_context_len,
            **solution_config,
        )
        self.ff = FitnessFeaturizer(
            popsize=self.popsize,
            num_dims=self.num_dims,
            seq_len=self.max_context_len,
            **fitness_config,
        )
        self.df = DistributionFeaturizer(
            popsize=self.popsize,
            num_dims=self.num_dims,
            seq_len=self.max_context_len,
            **distrib_config,
        )
        if net_ckpt_path is not None:
            self.ckpt = load_pkl_object(net_ckpt_path)
            self.default_net_params = self.ckpt["net_params"]
            self.model_config = self.ckpt["model_config"]
            self.model = EvoTransformer(**self.model_config)
            self.sf = SolutionFeaturizer(
                popsize=self.popsize,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["solution_config"],
            )
            self.ff = FitnessFeaturizer(
                popsize=self.popsize,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["fitness_config"],
            )
            self.df = DistributionFeaturizer(
                popsize=self.popsize,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["distrib_config"],
            )
            print(f"Loaded EvoTF model from ckpt: {net_ckpt_path}")

        if net_params is not None:
            self.default_net_params = net_params

        if net_params is None and net_ckpt_path is None:
            ckpt_fname = "2024_03_SNES_small.pkl"
            data = pkgutil.get_data(__name__, f"ckpt/evotf/{ckpt_fname}")
            self.ckpt = load_pkl_object(data, pkg_load=True)
            self.default_net_params = self.ckpt["net_params"]
            self.model_config = self.ckpt["model_config"]
            self.model = EvoTransformer(**self.model_config)
            self.sf = SolutionFeaturizer(
                popsize=self.popsize,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["solution_config"],
            )
            self.ff = FitnessFeaturizer(
                popsize=self.popsize,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["fitness_config"],
            )
            self.df = DistributionFeaturizer(
                popsize=self.popsize,
                num_dims=self.num_dims,
                seq_len=self.max_context_len,
                **self.ckpt["distrib_config"],
            )
            print(f"Loaded pretrained EvoTF model from ckpt: {ckpt_fname}")

        self.num_network_params = sum(
            x.size for x in jax.tree_leaves(self.default_net_params)
        )

        # Set core kwargs es_params
        self.sigma_init = sigma_init
        self.use_antithetic_sampling = use_antithetic_sampling
        if self.use_antithetic_sampling:
            assert not self.popsize & 1, "Population size must be even"
        # Setup look-ahead mask for forward pass
        self.la_mask = jnp.tril(jnp.ones((self.max_context_len, self.max_context_len)))
        self.strategy_name = "EvoTF_ES"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolutionary strategy."""
        params = EvoParams(
            sigma_init=self.sigma_init,
            net_params=self.default_net_params,
        )
        return params

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolutionary strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        scon_shape = (
            1,
            self.max_context_len,
            self.popsize,
            self.num_dims,
            self.sf.num_features,
        )
        fcon_shape = (
            1,
            self.max_context_len,
            self.popsize,
            self.ff.num_features,
        )
        dcon_shape = (
            1,
            self.max_context_len,
            self.num_dims,
            self.df.num_features,
        )
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init * jnp.ones(self.num_dims),
            sf_state=self.sf.initialize(),
            ff_state=self.ff.initialize(),
            df_state=self.df.initialize(),
            solution_context=jnp.zeros(scon_shape),
            fitness_context=jnp.zeros(fcon_shape),
            distribution_context=jnp.zeros(dcon_shape),
            best_member=initialization,
        )

        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        if not self.use_antithetic_sampling:
            noise = jax.random.normal(rng, (self.popsize, self.num_dims))
        else:
            noise_p = jax.random.normal(rng, (int(self.popsize / 2), self.num_dims))
            noise = jnp.concatenate([noise_p, -noise_p], axis=0)
        x = state.mean + noise * state.sigma.reshape(1, self.num_dims)
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # Get features from population info
        sfeatures, sf_state = self.sf.featurize(
            x, fitness, state.mean, state.sigma, state.sf_state
        )
        ffeatures, ff_state = self.ff.featurize(x, fitness, state.ff_state)
        dfeatures, df_state = self.df.featurize(
            x, fitness, state.mean, state.sigma, state.df_state
        )

        # Update the context with a sliding window
        shift_buffer = state.gen_counter >= self.max_context_len
        idx = jax.lax.select(shift_buffer, self.max_context_len - 1, state.gen_counter)
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
                params.net_params,
                solution_context,
                fitness_context,
                distribution_context,
                rngs={"dropout": jax.random.PRNGKey(0)},
                train=False,
                mask=self.la_mask,
            )

        pred, att = infer_step(solution_context, fitness_context, distribution_context)
        pred_mean = state.mean + params.lrate_mean * state.sigma * pred[0, 0, idx]
        pred_sigma = state.sigma * jnp.exp(params.lrate_sigma / 2 * pred[1, 0, idx])
        pred_sigma = jnp.clip(pred_sigma, 1e-08)
        # Collect and sort attention by population fitness order
        return state.replace(
            mean=pred_mean,
            sigma=pred_sigma,
            sf_state=sf_state,
            ff_state=ff_state,
            df_state=df_state,
            solution_context=solution_context,
            fitness_context=fitness_context,
            distribution_context=distribution_context,
        )
