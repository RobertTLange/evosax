"""Abstract class for optimization problems."""

from functools import partial

import jax
from flax import struct

from evosax.types import Fitness, Metrics, Population, Solution


@struct.dataclass
class State:
    counter: int


class Problem:
    """Abstract class for optimization problems."""

    @property
    def num_dims(self) -> int:
        """Number of dimensions of the problem."""
        solution = self.sample(jax.random.key(0))
        return sum(x.size for x in jax.tree.leaves(solution))

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array) -> State:
        """Initialize state."""
        return State(counter=0)

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self,
        key: jax.Array,
        solutions: Population,
        state: State,
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a batch of solutions."""
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        raise NotImplementedError
