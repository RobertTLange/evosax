"""Type definitions for evosax."""

from typing import Any, TypeAlias

import jax

ArrayTree: TypeAlias = Any

Solution: TypeAlias = ArrayTree
Population: TypeAlias = ArrayTree
Fitness: TypeAlias = jax.Array
