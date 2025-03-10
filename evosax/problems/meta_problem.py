"""Abstract class for meta-optimization problems."""

from functools import partial

import jax
from flax import struct

from evosax.types import Fitness, Metrics, Population

from .problem import Problem, State


@struct.dataclass
class Params:
    pass


@struct.dataclass
class State(State):
    pass


class MetaProblem(Problem):
    """Abstract class for meta-optimization problems."""

    @partial(jax.jit, static_argnames=("self",))
    def sample_params(self, key: jax.Array) -> Params:
        """Sample params for the meta-problem."""
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array, params: Params) -> State:
        """Initialize state of the meta-problem."""
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self,
        key: jax.Array,
        solutions: Population,
        state: State,
        params: Params,
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a batch of solutions."""
        raise NotImplementedError
