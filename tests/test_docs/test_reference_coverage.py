"""Tests that the API reference under ``docs/reference`` covers the public API."""

import inspect
import re
from pathlib import Path

import pytest
from evosax import problems
from evosax.algorithms import algorithms

REFERENCE_DIR = Path(__file__).parents[2] / "docs" / "reference"
DIRECTIVE = re.compile(r"^::: (\S+)$", re.MULTILINE)


@pytest.fixture(scope="module")
def documented() -> set[str]:
    """Identifiers of all ``:::`` directives in the API reference."""
    return {
        identifier
        for page in REFERENCE_DIR.rglob("*.md")
        for identifier in DIRECTIVE.findall(page.read_text())
    }


def is_documented(obj, documented: set[str]) -> bool:
    """Check for a directive on the object itself or on its defining module."""
    return (
        f"{obj.__module__}.{obj.__qualname__}" in documented
        or obj.__module__ in documented
    )


def test_every_algorithm_has_a_reference_page(documented):
    """Adding an algorithm to the registry requires adding a reference page."""
    missing = [
        cls.__name__
        for cls in algorithms.values()
        if not is_documented(cls, documented)
    ]
    assert missing == []


def test_every_public_problem_is_documented(documented):
    """Classes and functions exported by ``evosax.problems`` are documented."""
    exported = [getattr(problems, name) for name in problems.__all__]
    missing = [
        obj.__name__
        for obj in exported
        if (inspect.isclass(obj) or inspect.isfunction(obj))
        and not is_documented(obj, documented)
    ]
    assert missing == []
