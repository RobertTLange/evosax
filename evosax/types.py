"""Type definitions."""

from typing import Any, TypeAlias

import jax
from flax import struct

PyTree: TypeAlias = Any

Solution: TypeAlias = PyTree
Population: TypeAlias = PyTree
Fitness: TypeAlias = jax.Array
Metrics: TypeAlias = PyTree


@struct.dataclass
class State:
    pass


@struct.dataclass
class Params:
    pass
