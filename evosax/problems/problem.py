"""Abstract class for optimization problems."""

import jax

from ..types import Fitness, Population, PyTree, Solution


class Problem:
    """Abstract class for optimization problems."""

    def eval(self, key: jax.Array, solutions: Population) -> tuple[Fitness, PyTree]:
        """Evaluate a batch of solutions."""
        raise NotImplementedError

    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        raise NotImplementedError
