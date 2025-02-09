from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from .core import FitnessShaper
from .types import Fitness, Population, Solution
from .utils import get_best_fitness_member
from .utils.helpers import get_ravel_fn


@struct.dataclass
class State:
    mean: jax.Array
    sigma: float
    best_member: jax.Array
    best_fitness: float
    generation_counter: int


@struct.dataclass
class Params:
    sigma_init: float = 0.03
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class Strategy:
    def __init__(
        self,
        population_size: int,
        solution: Solution,
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Base Class for an Evolution Strategy."""
        self.population_size = population_size
        self.solution = solution

        self.ravel_solution, self.unravel_solution = get_ravel_fn(solution)
        flat_params = self.ravel_solution(solution)
        self.num_dims = flat_params.size

        # Mean exponential decay coefficient m' = coeff * m
        # Implements form of weight decay regularization
        self.mean_decay = mean_decay
        self.use_mean_decay = mean_decay > 0.0

        # Setup optional fitness shaper
        self.fitness_shaper = FitnessShaper(**fitness_kwargs)

    @property
    def default_params(self) -> Params:
        """Return default parameters of evolution strategy."""
        params = self.params_strategy
        return params

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        params: Params | None = None,
        init_solution: Solution | None = None,
    ) -> State:
        """`init` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Initialize strategy based on strategy-specific initialize method
        state = self.init_strategy(key, params)

        if init_solution is not None:
            state = self.set_mean(state, init_solution)

        return state

    @partial(jax.jit, static_argnames=("self",))
    def ask(
        self,
        key: jax.Array,
        state: State,
        params: Params | None = None,
    ) -> tuple[Population, State]:
        """`ask` for new parameter candidates to evaluate next."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Generate proposal based on strategy-specific ask method
        x, state = self.ask_strategy(key, state, params)
        # Clip proposal candidates into allowed range
        x_clipped = jnp.clip(x, params.clip_min, params.clip_max)

        # Unravel params
        x_out = jax.vmap(self.unravel_solution)(x_clipped)
        return x_out, state

    @partial(jax.jit, static_argnames=("self",))
    def tell(
        self,
        x: Population,
        fitness: Fitness,
        state: State,
        params: Params | None = None,
    ) -> State:
        """`tell` performance data for strategy state update."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # Ravel params
        x = jax.vmap(self.ravel_solution)(x)

        # Perform fitness reshaping inside of strategy tell call (if desired)
        fitness_re = self.fitness_shaper.apply(x, fitness)

        # Update the search state based on strategy-specific update
        state = self.tell_strategy(x, fitness_re, state, params)

        # Check if there is a new best member & update trackers
        best_member, best_fitness = get_best_fitness_member(
            x, fitness, state, self.fitness_shaper.maximize
        )

        # Exponentially decay mean if coefficient > 0.0
        if self.use_mean_decay:
            state = state.replace(mean=(1 - self.mean_decay) * state.mean)

        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
            generation_counter=state.generation_counter + 1,
        )

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """Strategy-specific `init` method. Returns initial state."""
        raise NotImplementedError

    def ask_strategy(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[jax.Array, State]:
        """Search-specific `ask` request. Returns proposals & updated state."""
        raise NotImplementedError

    def tell_strategy(
        self,
        x: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        """Search-specific `tell` update. Returns updated state."""
        raise NotImplementedError

    def get_eval_params(self, state: State):
        """Return reshaped parameters to evaluate."""
        x_out = self.unravel_solution(state.mean)
        return x_out

    def set_mean(self, state: State, solution: Solution) -> State:
        mean = self.ravel_solution(solution)
        state = state.replace(mean=mean)
        return state
