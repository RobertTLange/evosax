# Base classes

Every algorithm derives from `EvolutionaryAlgorithm` through one of two families.
The methods documented here make up the public interface of an algorithm: `init`, `ask`
and `tell`, the `default_params` property, and the accessors `get_mean` or
`get_best_solution` and `get_population`. The individual algorithm pages list the
constructor arguments.

::: evosax.algorithms.base.EvolutionaryAlgorithm

::: evosax.algorithms.distribution_based.base.DistributionBasedAlgorithm
    options:
      inherited_members: false

::: evosax.algorithms.population_based.base.PopulationBasedAlgorithm
    options:
      inherited_members: false

::: evosax.algorithms.distribution_based.sv.base.SV_ES
    options:
      inherited_members: false

## State and metrics

::: evosax.algorithms.base.State
    options:
      heading_level: 3

::: evosax.algorithms.base.metrics_fn
    options:
      heading_level: 3
