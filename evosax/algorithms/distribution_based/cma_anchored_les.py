"""CMA-anchored Learned Evolution Strategy with frozen meta-trained residuals.

A separable CMA-ES core with dimensionless learned gates, adaptive antithetic
sampling, pair-consistent evolution paths, and capped stall re-expansion.
Checkpoint provenance is packaged beside the frozen NumPy pickle weights.
"""

import math
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.special import erfinv

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, PyTree, Solution

from ...learned_evolution.cma_anchored_les_tools import (
    ANTITHETIC_SAMPLING,
    C_MEAN,
    EPS,
    IMPROVER_BUDGETED_MIRRORING,
    INNER_STD_INIT,
    LOG_AXIS_BOUND,
    LOG_AXIS_STEP_BOUND,
    LOG_SIGMA_STEP_BOUND,
    PATH_PAIR_CORRECTION,
    RESTART_EXPANSION,
    SIGMA_MAX_FACTOR,
    SIGMA_MIN_FACTOR,
    SUCCESS_RATES,
    DimensionGate,
    GlobalGates,
    RecombinationResidual,
    bounded_gate,
    cma_constants,
    fitness_ranks,
    load_parameters,
)
from ..base import update_best_solution_and_fitness
from .base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)


@struct.dataclass
class State(BaseState):
    std: jax.Array
    sigma: jax.Array
    axis_std: jax.Array
    p_sigma: jax.Array
    p_c: jax.Array
    success_ema: jax.Array
    epoch_best_fitness: float
    stall_counter: jax.Array
    best_solution_shaped: jax.Array
    best_fitness_shaped: float


@struct.dataclass
class Params(BaseParams):
    std_init: float
    params: PyTree


class CMAAnchoredLES(DistributionBasedAlgorithm):
    """Separable CMA-ES with bounded residuals and bundled meta-trained weights."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        params: PyTree | None = None,
        params_path: str | None = None,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize with weights, a trusted pickle path, or bundled weights."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)
        if self.num_dims == 0:
            raise ValueError("CMAAnchoredLES requires a nonempty solution")
        self.recombination = RecombinationResidual()
        self.global_gates = GlobalGates()
        self.dimension_gate = DimensionGate()
        self.learned_params = load_parameters(params, params_path)
        constants = cma_constants(self.num_dims, population_size)
        self.base_weights = constants["base_weights"]
        self.c_sigma = constants["c_sigma"]
        self.d_sigma = constants["d_sigma"]
        self.c_c = constants["c_c"]
        self.c_1 = constants["c_1"]
        self.c_mu = constants["c_mu"]
        self.chi_d = constants["chi_d"]
        self.stall_patience = 120.0 + 30.0 * float(self.num_dims) / float(
            population_size
        )
        self._pair_base = (population_size + 1) // 2
        self._pair_count = population_size - self._pair_base

    @property
    def _default_params(self) -> Params:
        return Params(std_init=INNER_STD_INIT, params=self.learned_params)

    def _init(self, key: jax.Array, params: Params) -> State:
        del key
        sigma = jnp.asarray(params.std_init, dtype=jnp.float32)
        axis_std = jnp.ones(self.num_dims)
        return State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=sigma * axis_std,
            sigma=sigma,
            axis_std=axis_std,
            p_sigma=jnp.zeros(self.num_dims),
            p_c=jnp.zeros(self.num_dims),
            success_ema=jnp.full((SUCCESS_RATES.size,), 0.5),
            epoch_best_fitness=jnp.inf,
            stall_counter=jnp.zeros(()),
            best_solution_shaped=jnp.full((self.num_dims,), jnp.nan),
            best_fitness_shaped=jnp.inf,
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    def _mirror_gate(self, success_ema: jax.Array) -> jax.Array:
        """Fraction of the mirrorable pairs that is actually mirrored."""
        if IMPROVER_BUDGETED_MIRRORING:
            expected_improvers = float(self.population_size) * success_ema[0]
            return jnp.clip(
                1.0 - expected_improvers / float(max(self._pair_count, 1)), 0.0, 1.0
            )
        return jnp.asarray(1.0)

    def _mirror_mask(self, success_ema: jax.Array) -> jax.Array:
        """Boolean vector of static length ``_pair_count``: which pairs mirror."""
        return jnp.arange(self._pair_count) < self._mirror_gate(success_ema) * float(
            self._pair_count
        )

    def _ask(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[Population, State]:
        del params
        if ANTITHETIC_SAMPLING:
            (base_key, alt_key) = jax.random.split(key)
            base = jax.random.normal(base_key, (self._pair_base, self.num_dims))
            alt = jax.random.normal(alt_key, (self._pair_count, self.num_dims))
            mask = self._mirror_mask(state.success_ema)
            tail = jnp.where(mask[:, None], -base[: self._pair_count], alt)
            noise = jnp.concatenate((base, tail), axis=0)
        else:
            noise = jax.random.normal(key, (self.population_size, self.num_dims))
        return (state.mean + state.sigma * state.axis_std[None, ...] * noise, state)

    def _member_features(
        self,
        ranks: jax.Array,
        base_weights: jax.Array,
        noise: jax.Array,
        improved: jax.Array,
        p_sigma: jax.Array,
    ) -> jax.Array:
        """Per-member, order-statistic-only, dimensionless recombination inputs."""
        population_size = self.population_size
        dims = float(self.num_dims)
        rank_values = ranks.astype(jnp.float32)
        centered_rank = rank_values / max(population_size - 1, 1) - 0.5
        quantile = (
            jnp.clip(
                math.sqrt(2.0)
                * erfinv(2.0 * (rank_values + 0.5) / population_size - 1.0),
                -3.0,
                3.0,
            )
            / 2.0
        )
        radial = jnp.tanh(jnp.sum(jnp.square(noise), axis=-1) / dims - 1.0)
        base_weight_feature = jnp.tanh(population_size * base_weights - 1.0)
        path_norm = jnp.linalg.norm(p_sigma)
        noise_norm = jnp.linalg.norm(noise, axis=-1)
        cosine = jnp.sum(noise * p_sigma[None, ...], axis=-1) / (
            noise_norm * path_norm + EPS
        )
        alignment = jnp.tanh(math.sqrt(dims) * cosine)
        return jnp.stack(
            (
                centered_rank,
                quantile,
                improved - 0.5,
                radial,
                base_weight_feature,
                alignment,
            ),
            axis=-1,
        )

    def _global_features(
        self,
        csa_signal: jax.Array,
        p_sigma: jax.Array,
        p_c: jax.Array,
        axis_std: jax.Array,
        weights: jax.Array,
        noise: jax.Array,
        success_ema: jax.Array,
        success_now: jax.Array,
    ) -> jax.Array:
        """Global, dimensionless evidence for the step-size and mean gates."""
        dims = float(self.num_dims)
        rms_axis = jnp.sqrt(jnp.mean(jnp.square(axis_std)))
        rank_one_path = p_c / axis_std
        path_ratio = jnp.tanh(
            jnp.linalg.norm(p_c) / (self.chi_d * rms_axis + EPS) - 1.0
        )
        elite_radial = jnp.tanh(
            jnp.sum(weights * jnp.sum(jnp.square(noise), axis=-1)) / dims - 1.0
        )
        spread = jnp.tanh(jnp.std(jnp.log(axis_std)))
        alignment = jnp.tanh(
            math.sqrt(dims)
            * jnp.sum(p_sigma * rank_one_path)
            / (jnp.linalg.norm(p_sigma) * jnp.linalg.norm(rank_one_path) + EPS)
        )
        return jnp.concatenate(
            (
                jnp.stack(
                    (jnp.tanh(csa_signal), path_ratio, elite_radial, spread, alignment)
                ),
                2.0 * success_ema - 1.0,
                jnp.reshape(2.0 * success_now - 1.0, (1,)),
            )
        )

    def _dimension_features(
        self,
        csa_signal: jax.Array,
        p_sigma: jax.Array,
        p_c: jax.Array,
        axis_std: jax.Array,
        rank_mu_term: jax.Array,
        rank_one_term: jax.Array,
    ) -> jax.Array:
        """Per-dimension, normalized path and covariance evidence."""
        log_axis = jnp.log(axis_std)
        centered_log_axis = log_axis - jnp.mean(log_axis)
        broadcast_signal = jnp.full(axis_std.shape, jnp.tanh(csa_signal))
        return jnp.stack(
            (
                jnp.tanh(p_sigma),
                jnp.tanh(p_c / axis_std),
                jnp.tanh(centered_log_axis),
                jnp.tanh(rank_mu_term - 1.0),
                jnp.tanh(rank_one_term - 1.0),
                broadcast_signal,
            ),
            axis=-1,
        )

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        del key
        learned = params.params
        ranks = fitness_ranks(fitness)
        base_weights = self.base_weights[ranks]
        improved = (fitness < state.best_fitness_shaped).astype(jnp.float32)
        success_now = jnp.mean(improved)
        # Match the sampling law before advancing the success EMA.
        mirror_mask = self._mirror_mask(state.success_ema)
        success_ema = (
            1.0 - SUCCESS_RATES
        ) * state.success_ema + SUCCESS_RATES * success_now
        differences = (population - state.mean) / state.sigma
        noise = differences / state.axis_std[None, ...]
        member_features = self._member_features(
            ranks, base_weights, noise, improved, state.p_sigma
        )
        residual = cast(
            jax.Array,
            self.recombination.apply(learned["recombination"], member_features),
        )
        weights = base_weights * bounded_gate(residual)
        weights = weights / jnp.sum(weights)
        mu_eff = 1.0 / jnp.sum(jnp.square(weights))
        if ANTITHETIC_SAMPLING and PATH_PAIR_CORRECTION:
            pair_base = self._pair_base
            pair_count = self._pair_count
            w_a = weights[:pair_count]
            w_b = weights[pair_base : pair_base + pair_count]
            paired = jnp.where(
                mirror_mask, jnp.square(w_a - w_b), jnp.square(w_a) + jnp.square(w_b)
            )
            unpaired = weights[pair_count:pair_base]
            pair_mass = jnp.sum(paired) + jnp.sum(jnp.square(unpaired))
            mu_eff_path = jnp.minimum(
                1.0 / (pair_mass + EPS), float(self.population_size)
            )
        else:
            mu_eff_path = mu_eff
        weighted_difference = jnp.sum(weights[..., None] * differences, axis=0)
        weighted_noise = jnp.sum(weights[..., None] * noise, axis=0)
        p_sigma = (1.0 - self.c_sigma) * state.p_sigma + jnp.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * mu_eff_path
        ) * weighted_noise
        p_c = (1.0 - self.c_c) * state.p_c + jnp.sqrt(
            self.c_c * (2.0 - self.c_c) * mu_eff_path
        ) * weighted_difference
        csa_signal = jnp.linalg.norm(p_sigma) / self.chi_d - 1.0
        gates = cast(
            jax.Array,
            self.global_gates.apply(
                learned["global_gates"],
                self._global_features(
                    csa_signal,
                    p_sigma,
                    p_c,
                    state.axis_std,
                    weights,
                    noise,
                    success_ema,
                    success_now,
                ),
            ),
        )
        sigma_gate = bounded_gate(gates[0])
        mean_gate = bounded_gate(gates[1])
        mean = state.mean + C_MEAN * mean_gate * state.sigma * weighted_difference
        rank_mu_term = jnp.sum(weights[..., None] * jnp.square(noise), axis=0)
        rank_one_term = jnp.square(p_c / state.axis_std)
        ratio = (
            1.0 + self.c_1 * (rank_one_term - 1.0) + self.c_mu * (rank_mu_term - 1.0)
        )
        ratio = jnp.clip(ratio, 1e-08, 100000000.0)
        axis_gate = bounded_gate(
            cast(
                jax.Array,
                self.dimension_gate.apply(
                    learned["dimension_gate"],
                    self._dimension_features(
                        csa_signal,
                        p_sigma,
                        p_c,
                        state.axis_std,
                        rank_mu_term,
                        rank_one_term,
                    ),
                ),
            )
        )
        log_axis = jnp.log(state.axis_std) + jnp.clip(
            0.5 * axis_gate * jnp.log(ratio), -LOG_AXIS_STEP_BOUND, LOG_AXIS_STEP_BOUND
        )
        log_sigma_step = jnp.clip(
            self.c_sigma / self.d_sigma * csa_signal * sigma_gate,
            -LOG_SIGMA_STEP_BOUND,
            LOG_SIGMA_STEP_BOUND,
        )
        # Preserve sigma * axis_std; p_c uses the same axis units.
        shift = jnp.mean(log_axis)
        log_axis = jnp.clip(log_axis - shift, -LOG_AXIS_BOUND, LOG_AXIS_BOUND)
        axis_std = jnp.exp(log_axis)
        p_c = p_c * jnp.exp(-shift)
        sigma = state.sigma * jnp.exp(shift + log_sigma_step)
        sigma = jnp.clip(
            sigma,
            params.std_init * SIGMA_MIN_FACTOR,
            params.std_init * SIGMA_MAX_FACTOR,
        )
        (best_solution_shaped, best_fitness_shaped) = update_best_solution_and_fitness(
            population, fitness, state.best_solution_shaped, state.best_fitness_shaped
        )
        generation_best = jnp.min(fitness)
        epoch_improved = generation_best < state.epoch_best_fitness
        epoch_best_fitness = jnp.minimum(state.epoch_best_fitness, generation_best)
        stall_counter = jnp.where(epoch_improved, 0.0, state.stall_counter + 1.0)
        fire = stall_counter >= self.stall_patience
        sigma = jnp.where(
            fire,
            jnp.maximum(sigma, jnp.minimum(sigma * RESTART_EXPANSION, params.std_init)),
            sigma,
        )
        # Re-anchor without resetting the monotone global incumbent.
        mean = jnp.where(fire, state.best_solution_shaped, mean)
        axis_std = jnp.where(fire, jnp.ones(self.num_dims), axis_std)
        p_sigma = jnp.where(fire, jnp.zeros(self.num_dims), p_sigma)
        p_c = jnp.where(fire, jnp.zeros(self.num_dims), p_c)
        success_ema = jnp.where(fire, jnp.full((SUCCESS_RATES.size,), 0.5), success_ema)
        epoch_best_fitness = jnp.where(fire, jnp.inf, epoch_best_fitness)
        stall_counter = jnp.where(fire, 0.0, stall_counter)
        return state.replace(
            mean=mean,
            std=sigma * axis_std,
            sigma=sigma,
            axis_std=axis_std,
            p_sigma=p_sigma,
            p_c=p_c,
            success_ema=success_ema,
            epoch_best_fitness=epoch_best_fitness,
            stall_counter=stall_counter,
            best_solution_shaped=best_solution_shaped,
            best_fitness_shaped=best_fitness_shaped,
        )
