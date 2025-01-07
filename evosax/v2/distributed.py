import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional, Union, Any
from flax import struct
from evosax.core import ParameterReshaper, FitnessShaper


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: chex.Array
    gen_counter: int


@struct.dataclass
class EvoParams:
    clip_min: float
    clip_max: float
    sigma_init: float = 0.03
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0


@struct.dataclass
class EvoUpdate:
    delta_mean: chex.Array
    delta_sigma: chex.Array


class DistributedStrategy(object):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        mean_decay: float = 0.0,
        n_devices: int = 1,
        param_dtype: Any = jnp.float32,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Base Class for an Evolution Strategy."""
        self.total_popsize = popsize
        self.n_devices = n_devices
        self.popsize = self.total_popsize // self.n_devices
        self.param_dtype = param_dtype

        # Setup optional parameter reshaper
        self.use_param_reshaper = pholder_params is not None
        if self.use_param_reshaper:
            self.param_reshaper = ParameterReshaper(pholder_params, n_devices=1)
            self.num_dims = self.param_reshaper.total_params
        else:
            self.num_dims = num_dims
        assert (
            self.num_dims is not None
        ), "Provide either num_dims or pholder_params to strategy."

        # Mean exponential decay coefficient m' = coeff * m
        # Implements form of weight decay regularization
        self.mean_decay = mean_decay
        self.use_mean_decay = mean_decay > 0.0

        # Setup optional fitness shaper
        self.fitness_shaper = FitnessShaper(**fitness_kwargs)

    @property
    def default_params(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        params = self.params_strategy
        return params

    def initialize(
        self,
        rng: chex.PRNGKey,
        params: Optional[EvoParams] = None,
        init_mean: Optional[Union[chex.Array, chex.ArrayTree]] = None,
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Initialize strategy based on strategy-specific initialize method
        state = self.initialize_strategy(rng, params)

        if init_mean is not None:
            state = self.set_mean(state, init_mean)
        return state

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """Search-specific `initialize` method. Returns initial state."""
        raise NotImplementedError

    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> Tuple[Union[chex.Array, chex.ArrayTree], EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Generate proposal based on strategy-specific ask method
        x, state = self.ask_strategy(rng, state, params)
        # Clip proposal candidates into allowed range
        x_clipped = jnp.clip(x, params.clip_min, params.clip_max)

        # Reshape parameters into pytrees
        if self.use_param_reshaper:
            x_out = self.param_reshaper.reshape(x_clipped)
        else:
            x_out = x_clipped
        return x_out, state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """Search-specific `ask` request. Returns proposals & updated state."""
        raise NotImplementedError

    def tell(
        self,
        x: Union[chex.Array, chex.ArrayTree],
        fitness: chex.Array,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Flatten params if using param reshaper for ES update
        if self.use_param_reshaper:
            x = self.param_reshaper.flatten(x)

        # Perform fitness reshaping inside of strategy tell call (if desired)
        fitness_re = self.fitness_shaper.apply(x, fitness)

        # Update the search state based on strategy-specific update
        update = self.get_update(x, fitness_re, state, params)
        state = self.apply_update(update, state, params)
        return state

    def get_update(
        self,
        x: Union[chex.Array, chex.ArrayTree],
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoUpdate:
        """Get strategy state update (per device-level)."""
        # Update the search state based on strategy-specific update
        evo_update = self.get_update_strategy(x, fitness, state, params)
        return evo_update

    def get_update_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoUpdate:
        """Search-specific `tell` update computation. Returns state update."""
        raise NotImplementedError

    def apply_update(
        self,
        update: EvoUpdate,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> EvoState:
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params
        # Exponentially decay mean if coefficient > 0.0
        state = self.apply_update_strategy(update, state, params)
        if self.use_mean_decay:
            state = state.replace(mean=(1 - self.mean_decay) * state.mean)
        return state.replace(
            gen_counter=state.gen_counter + 1,
        )

    def apply_update_strategy(
        self,
        update: EvoUpdate,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """Search-specific `tell` update application. Returns updated state."""
        raise NotImplementedError

    def get_eval_params(self, state: EvoState):
        """Return reshaped parameters to evaluate."""
        mean_single = state.mean[0]

        if self.use_param_reshaper:
            x_out = self.param_reshaper.reshape_single(mean_single)
        else:
            x_out = mean_single
        return x_out

    def set_mean(
        self, state: EvoState, replace_mean: Union[chex.Array, chex.ArrayTree]
    ) -> EvoState:
        if self.use_param_reshaper:
            replace_mean = self.param_reshaper.flatten_single(replace_mean)
        else:
            replace_mean = jnp.asarray(replace_mean)
        state = state.replace(mean=replace_mean)
        return state
