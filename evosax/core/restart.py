"""Restart utilities for distribution-based Evolution Strategies."""

from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias

import jax
import jax.numpy as jnp
from flax import serialization, struct

from evosax.types import Fitness, Params, Population, State

from ..algorithms.distribution_based.base import DistributionBasedAlgorithm
from ..algorithms.distribution_based.cma_es import (
    Params as CMAESParams,
    eigen_decomposition,
)

RestartCondition: TypeAlias = Callable[
    [Population, Fitness, State, Params, "RestartState", "RestartParams"], jax.Array
]
StrategyFactory: TypeAlias = Callable[[int], DistributionBasedAlgorithm]
StrategyParamsFactory: TypeAlias = Callable[
    [DistributionBasedAlgorithm, Params], Params
]
SmallPopulationParamsFactory: TypeAlias = Callable[
    [jax.Array, DistributionBasedAlgorithm, Params, float], Params
]
PopulationSizeTransform: TypeAlias = Callable[[int], int]


class GenerationRestartState(Protocol):
    generation_counter: int


class GenerationRestartParams(Protocol):
    generation_threshold: int


class SpreadRestartParams(Protocol):
    min_fitness_spread: float


class CMAESRestartState(Protocol):
    C: jax.Array
    mean: jax.Array
    std: float | jax.Array
    p_c: jax.Array


class CMAESRestartParams(Protocol):
    tol_x: float
    tol_x_up: float
    tol_condition_C: float


class AmalgamRestartState(Protocol):
    c_mult: jax.Array


@struct.dataclass
class RestartState:
    restart_counter: int


@struct.dataclass
class RestartParams:
    min_num_gens: int = 0
    min_fitness_spread: float = 1e-12
    copy_mean: bool = False
    generation_threshold: int = 2**31 - 1
    tol_x: float = 1e-12
    tol_x_up: float = 1e4
    tol_condition_C: float = 1e14
    tol: float = 0.001
    atol: float = 0.0


@struct.dataclass
class FitnessStdRestartParams(RestartParams):
    pass


def generation_cond(
    population: Population,
    fitness: Fitness,
    state: GenerationRestartState,
    params: Params,
    restart_state: RestartState,
    restart_params: GenerationRestartParams,
) -> bool:
    """Stop after a certain number of generations."""
    return state.generation_counter >= restart_params.generation_threshold


def spread_cond(
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
    restart_state: RestartState,
    restart_params: SpreadRestartParams,
) -> jax.Array:
    """Stop if fitness max minus fitness min is below threshold."""
    return jnp.max(fitness) - jnp.min(fitness) < restart_params.min_fitness_spread


def fitness_std_cond(
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
    restart_state: RestartState,
    restart_params: FitnessStdRestartParams,
) -> jax.Array:
    """Stop if fitness standard deviation is below tolerance."""
    finite = jnp.all(jnp.isfinite(fitness))
    threshold = restart_params.atol + restart_params.tol * jnp.abs(jnp.mean(fitness))
    return jnp.logical_and(finite, jnp.std(fitness) <= threshold)


def cma_cond(
    population: Population,
    fitness: Fitness,
    state: CMAESRestartState,
    params: Params,
    restart_state: RestartState,
    restart_params: CMAESRestartParams,
) -> jax.Array:
    """Stop if condition specific to CMA-ES is met.

    Default tolerances:
    tol_x: 1e-12 * sigma
    tol_x_up: 1e4
    tol_condition_C: 1e14
    """
    dC = jnp.diag(state.C)
    _, B, D = eigen_decomposition(state.C)

    # Stop if std of normal distribution is smaller than tolx in all coordinates
    # and pc is smaller than tolx in all components.
    cond_s_1 = jnp.all(state.std * jnp.sqrt(dC) < restart_params.tol_x)
    cond_s_2 = jnp.all(jnp.abs(state.std * state.p_c) < restart_params.tol_x)
    cond_1 = jnp.logical_and(cond_s_1, cond_s_2)

    # Stop if std diverges
    cond_2 = state.std * jnp.max(D) > restart_params.tol_x_up

    # Stop if adding 0.2 std does not change mean.
    cond_no_coord_change = jnp.any(
        state.mean == state.mean + (0.2 * state.std * jnp.sqrt(dC))
    )
    cond_3 = cond_no_coord_change

    # Stop if adding 0.1 std in principal directions of C does not change mean.
    cond_no_axis_change = jnp.all(
        state.mean == state.mean + (0.1 * state.std * D[0] * B[:, 0])
    )
    cond_4 = cond_no_axis_change

    # Stop if the condition number of the covariance matrix exceeds 1e14.
    condition_number = (jnp.max(D) / jnp.min(D)) ** 2
    cond_condition_cov = condition_number > restart_params.tol_condition_C
    cond_5 = cond_condition_cov

    return cond_1 | cond_2 | cond_3 | cond_4 | cond_5


def amalgam_cond(
    population: Population,
    fitness: Fitness,
    state: AmalgamRestartState,
    params: Params,
    restart_state: RestartState,
    restart_params: RestartParams,
) -> jax.Array:
    """Stop if c_mult is below threshold."""
    return state.c_mult < 1e-10


@struct.dataclass
class IPOPRestartState(RestartState):
    restart_counter: int
    restart_next: bool
    active_population_size: int


@struct.dataclass
class IPOPRestartParams(RestartParams):
    min_num_gens: int = 50
    population_size_multiplier: int = 2


@struct.dataclass
class BIPOPRestartState(RestartState):
    restart_counter: int
    restart_next: bool
    active_population_size: int
    restart_large_counter: int
    large_eval_budget: int
    small_eval_budget: int
    small_pop_active: bool
    initial_std: float = float("nan")


@struct.dataclass
class BIPOPRestartParams(RestartParams):
    min_num_gens: int = 50
    population_size_multiplier: int = 2


def _bipop_restart_state_to_state_dict(
    restart_state: BIPOPRestartState,
) -> dict[str, object]:
    return {
        name: serialization.to_state_dict(getattr(restart_state, name))
        for name in BIPOPRestartState.__dataclass_fields__
    }


def _bipop_restart_state_from_state_dict(
    restart_state: BIPOPRestartState,
    state_dict: dict[str, object],
) -> BIPOPRestartState:
    state_dict = state_dict.copy()
    state_dict.setdefault("initial_std", restart_state.initial_std)
    restored_state = restart_state.replace(
        **{
            name: serialization.from_state_dict(
                getattr(restart_state, name), state_dict.pop(name), name=name
            )
            for name in BIPOPRestartState.__dataclass_fields__
        }
    )
    if state_dict:
        names = ",".join(state_dict)
        raise ValueError(
            f'Unknown field(s) "{names}" in state dict while restoring '
            f"BIPOPRestartState at path {serialization.current_path()}"
        )
    return restored_state


serialization.register_serialization_state(
    BIPOPRestartState,
    _bipop_restart_state_to_state_dict,
    _bipop_restart_state_from_state_dict,
    override=True,
)


class RestartController:
    """Evaluate stop criteria for a single-distribution strategy."""

    def __init__(
        self,
        stop_criteria: Sequence[RestartCondition] = (spread_cond,),
        restart_params: RestartParams | None = None,
    ):
        if not stop_criteria:
            raise ValueError("At least one restart criterion is required.")

        self.stop_criteria = tuple(stop_criteria)
        self.restart_params = restart_params or RestartParams()

    def should_restart(
        self,
        population: Population,
        fitness: Fitness,
        state: State,
        strategy_params: Params,
        restart_state: RestartState,
    ) -> jax.Array:
        """Return whether a completed generation should trigger a restart."""
        _validate_single_distribution_state(state)
        criteria_met = jnp.asarray(
            [
                criterion(
                    population,
                    fitness,
                    state,
                    strategy_params,
                    restart_state,
                    self.restart_params,
                )
                for criterion in self.stop_criteria
            ]
        )
        min_generations_met = (
            state.generation_counter >= self.restart_params.min_num_gens
        )
        return jnp.logical_and(min_generations_met, jnp.any(criteria_met))


class SimpleRestart(RestartController):
    """Reinitialize a distribution-based strategy with a fixed population size."""

    def init(self) -> RestartState:
        """Create restart state for a new optimization run."""
        return RestartState(restart_counter=0)

    def restart(
        self,
        key: jax.Array,
        strategy: DistributionBasedAlgorithm,
        state: State,
        strategy_params: Params,
        restart_state: RestartState,
    ) -> tuple[State, RestartState]:
        """Reinitialize a strategy while preserving its run archive."""
        _validate_single_distribution_state(state)
        restarted_state = _restart_strategy(
            key,
            strategy,
            strategy,
            state,
            strategy_params,
            self.restart_params,
        )
        return restarted_state, restart_state.replace(
            restart_counter=restart_state.restart_counter + 1
        )


class IPOPRestart(RestartController):
    """Increase the population size after every restart.

    ``strategy_params_factory`` maps parameters from the previous strategy to a
    rebuilt one. ``population_size_transform`` adapts requested sizes to strategy
    constraints such as even antithetic populations.
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        initial_population_size: int,
        stop_criteria: Sequence[RestartCondition] = (spread_cond,),
        restart_params: IPOPRestartParams | None = None,
        strategy_params_factory: StrategyParamsFactory | None = None,
        population_size_transform: PopulationSizeTransform | None = None,
    ):
        resolved_restart_params = restart_params or IPOPRestartParams()
        super().__init__(stop_criteria, resolved_restart_params)
        self.restart_params: IPOPRestartParams = resolved_restart_params
        self.strategy_factory = strategy_factory
        self.initial_population_size = initial_population_size
        self.strategy_params_factory = (
            strategy_params_factory or _default_strategy_params
        )
        self.population_size_transform = (
            population_size_transform or _identity_population_size
        )

    def init(self, strategy: DistributionBasedAlgorithm) -> IPOPRestartState:
        """Create restart state for an IPOP run."""
        _validate_initial_population_size(strategy, self.initial_population_size)
        return IPOPRestartState(
            restart_counter=0,
            restart_next=False,
            active_population_size=self.initial_population_size,
        )

    def restart(
        self,
        key: jax.Array,
        strategy: DistributionBasedAlgorithm,
        state: State,
        strategy_params: Params,
        restart_state: IPOPRestartState,
    ) -> tuple[DistributionBasedAlgorithm, Params, State, IPOPRestartState]:
        """Rebuild a strategy with an increased population size."""
        _validate_single_distribution_state(state)
        next_population_size = int(
            restart_state.active_population_size
            * self.restart_params.population_size_multiplier
        )
        next_strategy = self.strategy_factory(
            self.population_size_transform(next_population_size)
        )
        next_strategy_params = self.strategy_params_factory(
            next_strategy, strategy_params
        )
        next_state = _restart_strategy(
            key,
            strategy,
            next_strategy,
            state,
            next_strategy_params,
            self.restart_params,
        )
        next_restart_state = restart_state.replace(
            restart_counter=restart_state.restart_counter + 1,
            restart_next=False,
            active_population_size=next_strategy.population_size,
        )
        return next_strategy, next_strategy_params, next_state, next_restart_state


class BIPOPRestart(RestartController):
    """Interleave small and large population restarts by evaluation budget.

    ``strategy_params_factory`` maps parameters from the previous strategy to a
    rebuilt one. ``population_size_transform`` adapts requested sizes to strategy
    constraints such as even antithetic populations. The optional
    ``small_population_params_factory`` configures small-regime restarts.
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        initial_population_size: int,
        stop_criteria: Sequence[RestartCondition] = (spread_cond,),
        restart_params: BIPOPRestartParams | None = None,
        strategy_params_factory: StrategyParamsFactory | None = None,
        population_size_transform: PopulationSizeTransform | None = None,
        small_population_params_factory: SmallPopulationParamsFactory | None = None,
    ):
        resolved_restart_params = restart_params or BIPOPRestartParams()
        super().__init__(stop_criteria, resolved_restart_params)
        self.restart_params: BIPOPRestartParams = resolved_restart_params
        self.strategy_factory = strategy_factory
        self.initial_population_size = initial_population_size
        self.strategy_params_factory = (
            strategy_params_factory or _default_strategy_params
        )
        self.population_size_transform = (
            population_size_transform or _identity_population_size
        )
        self.small_population_params_factory = (
            small_population_params_factory or _default_small_population_params
        )

    def init(self, strategy: DistributionBasedAlgorithm) -> BIPOPRestartState:
        """Create restart state for a BIPOP run."""
        _validate_initial_population_size(strategy, self.initial_population_size)
        return BIPOPRestartState(
            restart_counter=0,
            restart_next=False,
            active_population_size=self.initial_population_size,
            restart_large_counter=0,
            large_eval_budget=0,
            small_eval_budget=0,
            small_pop_active=True,
            initial_std=jnp.nan,
        )

    def restart(
        self,
        key: jax.Array,
        strategy: DistributionBasedAlgorithm,
        state: State,
        strategy_params: Params,
        restart_state: BIPOPRestartState,
    ) -> tuple[DistributionBasedAlgorithm, Params, State, BIPOPRestartState]:
        """Rebuild a strategy using the next BIPOP population size."""
        _validate_single_distribution_state(state)
        key_population, key_params, key_init = jax.random.split(key, 3)
        large_budget, small_budget = _updated_bipop_budgets(state, restart_state)
        use_small_population = small_budget < large_budget
        population_size, large_restart_counter = self._next_population_size(
            key_population, restart_state, use_small_population
        )
        next_strategy = self.strategy_factory(
            self.population_size_transform(population_size)
        )
        next_strategy_params = self.strategy_params_factory(
            next_strategy, strategy_params
        )
        initial_std = _initial_cma_std(strategy_params, restart_state)
        if use_small_population:
            next_strategy_params = self.small_population_params_factory(
                key_params, next_strategy, next_strategy_params, initial_std
            )
        next_state = _restart_strategy(
            key_init,
            strategy,
            next_strategy,
            state,
            next_strategy_params,
            self.restart_params,
        )
        next_restart_state = restart_state.replace(
            restart_counter=restart_state.restart_counter + 1,
            restart_next=False,
            active_population_size=next_strategy.population_size,
            restart_large_counter=large_restart_counter,
            large_eval_budget=large_budget,
            small_eval_budget=small_budget,
            small_pop_active=use_small_population,
            initial_std=initial_std,
        )
        return next_strategy, next_strategy_params, next_state, next_restart_state

    def _next_population_size(
        self,
        key: jax.Array,
        restart_state: BIPOPRestartState,
        use_small_population: bool,
    ) -> tuple[int, int]:
        if not use_small_population:
            next_large_population_multiplier = (
                self.restart_params.population_size_multiplier
                ** (restart_state.restart_large_counter + 1)
            )
            large_population_size = (
                self.initial_population_size * next_large_population_multiplier
            )
            return large_population_size, restart_state.restart_large_counter + 1

        exponent = float(jax.random.uniform(key) ** 2)
        current_large_population_multiplier = (
            self.restart_params.population_size_multiplier
            ** restart_state.restart_large_counter
        )
        small_population_multiplier = 0.5 * current_large_population_multiplier
        small_population_size = int(
            self.initial_population_size * small_population_multiplier**exponent
        )
        return max(1, small_population_size), restart_state.restart_large_counter


def _restart_strategy(
    key: jax.Array,
    previous_strategy: DistributionBasedAlgorithm,
    next_strategy: DistributionBasedAlgorithm,
    previous_state: State,
    next_params: Params,
    restart_params: RestartParams,
) -> State:
    restart_mean = (
        previous_strategy.get_mean(previous_state)
        if restart_params.copy_mean
        else next_strategy.solution
    )
    restarted_state = next_strategy.init(key, restart_mean, next_params)
    return restarted_state.replace(
        best_solution=previous_state.best_solution,
        best_fitness=previous_state.best_fitness,
    )


def _default_strategy_params(
    strategy: DistributionBasedAlgorithm, previous_params: Params
) -> Params:
    """Use population-shape-compatible defaults for a rebuilt strategy."""
    del previous_params
    return strategy.default_params


def _default_small_population_params(
    key: jax.Array,
    strategy: DistributionBasedAlgorithm,
    params: Params,
    initial_std: float,
) -> Params:
    """Apply BIPOP's randomized initial standard deviation for CMA-ES."""
    del strategy
    if isinstance(params, CMAESParams):
        std_init = initial_std * 10 ** (-2 * jax.random.uniform(key))
        return params.replace(std_init=std_init)
    return params


def _initial_cma_std(params: Params, restart_state: BIPOPRestartState) -> float:
    if isinstance(params, CMAESParams) and (
        restart_state.restart_counter == 0 or jnp.isnan(restart_state.initial_std)
    ):
        return params.std_init
    return restart_state.initial_std


def _identity_population_size(population_size: int) -> int:
    """Keep population sizes unchanged when the strategy has no constraint."""
    return population_size


def _validate_initial_population_size(
    strategy: DistributionBasedAlgorithm, initial_population_size: int
) -> None:
    if strategy.population_size != initial_population_size:
        raise ValueError(
            "initial_population_size must match the initial strategy population size."
        )


def _validate_single_distribution_state(state: State) -> None:
    if jnp.ndim(state.mean) != 1 or jnp.ndim(state.generation_counter) != 0:
        raise ValueError(
            "Restart controllers support single-distribution states only. "
            "Batched distribution strategies require a dedicated restart policy."
        )


def _updated_bipop_budgets(
    state: State, restart_state: BIPOPRestartState
) -> tuple[int, int]:
    if restart_state.restart_counter == 0:
        return restart_state.large_eval_budget, restart_state.small_eval_budget

    evaluations = restart_state.active_population_size * int(state.generation_counter)
    if restart_state.small_pop_active:
        return (
            restart_state.large_eval_budget,
            restart_state.small_eval_budget + evaluations,
        )
    return (
        restart_state.large_eval_budget + evaluations,
        restart_state.small_eval_budget,
    )
