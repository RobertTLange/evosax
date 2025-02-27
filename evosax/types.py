"""Type definitions for evosax."""

from typing import Any, TypeAlias

import jax

PyTree: TypeAlias = Any

Solution: TypeAlias = PyTree
Population: TypeAlias = PyTree
Fitness: TypeAlias = jax.Array
Metrics: TypeAlias = PyTree
