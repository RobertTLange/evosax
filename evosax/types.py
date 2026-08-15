"""Type definitions."""

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
    def replace(self: StateT, **updates: Any) -> StateT:
        """Return a copy with the supplied fields replaced."""
        return replace(self, **updates)


@struct.dataclass
class Params:
    pass
