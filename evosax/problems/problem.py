"""Abstract class for optimization problems.

A problem evaluates a population of solutions and returns one fitness value per
member. Problems are stateful in the functional sense: `eval` takes and returns
an explicit state.
"""

from functools import partial

import jax
from flax import struct

from evosax.types import Fitness, Metrics, Population, Solution


@struct.dataclass
class State:
    """State shared by all problems.

    Attributes:
        counter: Number of completed `eval` calls.
    """

    counter: int


class Problem:
    """Abstract class for optimization problems.

    Subclasses implement `eval` to compute the fitness of a population and
    `sample` to draw a solution from the search space. `sample` is commonly used
    to obtain the example `solution` required by the algorithm constructors.
    """

    @property
    def num_dims(self) -> int:
        """Number of search dimensions, i.e. the total size of a solution."""
        solution = self.sample(jax.random.key(0))
        return sum(x.size for x in jax.tree.leaves(solution))

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array) -> State:
        """Initialize the problem state.

        Args:
            key: PRNG key.

        Returns:
            The initial problem state.
        """
        return State(counter=0)

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self,
        key: jax.Array,
        solutions: Population,
        state: State,
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a population of solutions.

        Args:
            key: PRNG key.
            solutions: Population to evaluate, a PyTree with a leading population
                axis on every leaf.
            state: Current problem state.

        Returns:
            A tuple `(fitness, state, info)` with one fitness value per population
            member, the updated problem state and a dictionary of additional
            information.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution from the search space.

        Args:
            key: PRNG key.

        Returns:
            A single solution PyTree.
        """
        raise NotImplementedError
