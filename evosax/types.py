"""Type definitions shared across evosax.

`Solution`, `Population` and `Metrics` are PyTrees. A `Population` has the same
structure as a `Solution` with an additional leading population axis on every
leaf. `Fitness` is an array with one value per population member.
"""

from dataclasses import replace
from typing import Any, TypeAlias, TypeVar

import jax
from flax import struct

PyTree: TypeAlias = Any
StateT = TypeVar("StateT", bound="State")

Solution: TypeAlias = PyTree
Population: TypeAlias = PyTree
Fitness: TypeAlias = jax.Array
Metrics: TypeAlias = PyTree


@struct.dataclass
class State:
    """Base class for immutable state dataclasses."""

    def replace(self: StateT, **updates: Any) -> StateT:
        """Return a copy with the supplied fields replaced."""
        return replace(self, **updates)


@struct.dataclass
class Params:
    """Base class for immutable hyperparameter dataclasses."""

    pass
