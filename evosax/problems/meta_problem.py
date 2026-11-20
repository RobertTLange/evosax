"""Abstract class for meta-optimization problems.

A meta-problem is a family of problems indexed by `Params`. Sampling new params
yields a new problem instance, which makes meta-problems suitable for
meta-training and for benchmarking across many task instances.
"""

from functools import partial

import jax
from flax import struct

from evosax.types import Fitness, Metrics, Population

from .problem import Problem, State as BaseState


@struct.dataclass
class Params:
    """Parameters identifying one problem instance of a meta-problem."""

    pass


@struct.dataclass
class State(BaseState):
    """State shared by all meta-problems."""

    pass


class MetaProblem(Problem):
    """Abstract class for meta-optimization problems.

    Unlike `Problem`, `init` and `eval` take an additional `params` argument that
    selects the problem instance. Use `sample_params` to draw an instance.
    """

    @partial(jax.jit, static_argnames=("self",))
    def sample_params(self, key: jax.Array) -> Params:
        """Sample the parameters of a problem instance.

        Args:
            key: PRNG key.

        Returns:
            Parameters identifying one problem instance.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array, params: Params) -> State:
        """Initialize the problem state for a problem instance.

        Args:
            key: PRNG key.
            params: Parameters of the problem instance.

        Returns:
            The initial problem state.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self,
        key: jax.Array,
        solutions: Population,
        state: State,
        params: Params,
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a population of solutions on a problem instance.

        Args:
            key: PRNG key.
            solutions: Population to evaluate, a PyTree with a leading population
                axis on every leaf.
            state: Current problem state.
            params: Parameters of the problem instance.

        Returns:
            A tuple `(fitness, state, info)` with one fitness value per population
            member, the updated problem state and a dictionary of additional
            information.
        """
        raise NotImplementedError
