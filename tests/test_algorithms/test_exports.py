"""Tests for the public exports of the algorithm packages."""

import importlib

import pytest

PACKAGES = [
    "evosax.algorithms",
    "evosax.algorithms.distribution_based",
    "evosax.algorithms.population_based",
]


@pytest.mark.parametrize("package_name", PACKAGES)
def test_all_names_resolve(package_name):
    """Every name in ``__all__`` must be an attribute of the package."""
    package = importlib.import_module(package_name)
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []


@pytest.mark.parametrize("package_name", PACKAGES)
def test_star_import(package_name):
    """A star import must not raise on registry aliases such as ``LES``."""
    namespace = {}
    exec(f"from {package_name} import *", namespace)


def test_aliased_algorithms_are_exported_by_class_name():
    """Algorithms registered under an alias are exported by their class name."""
    from evosax import algorithms
    from evosax.algorithms import distribution_based, population_based

    assert {"DiscoveredES", "LearnedES"} <= set(distribution_based.__all__)
    assert "LearnedGA" in population_based.__all__
    assert {"DiscoveredES", "LearnedES", "LearnedGA"} <= set(algorithms.__all__)
