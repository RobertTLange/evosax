"""Diffusion Evolution (Zhang et al., 2024).

[1] https://arxiv.org/abs/2410.02543
Note: This implementation is based on the reference implementation https://github.com/openai/diffusion-evolution.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import Params, PopulationBasedAlgorithm, State, metrics_fn


@struct.dataclass
class State(State):
    population: Population
    fitness: Fitness
    std: jax.Array
    latent_projection: jax.Array


@struct.dataclass
class Params(Params):
    std_m: float
    scale_factor: float
    alphas: jax.Array


def ddim_schedule(num_step: int) -> tuple[jax.Array, jax.Array]:
    """Generate DDIM schedule."""
    eps = 1e-4
    power = 1.0
    alphas = jnp.linspace(1 - eps, eps ** (2 / power), num_step) ** power
    return alphas


def cosine_schedule(num_step: int) -> tuple[jax.Array, jax.Array]:
    """Generate cosine schedule."""
    eps = 1e-3
    alphas = jnp.cos(jnp.linspace(0, jnp.pi, num_step)) + 1
    alphas = alphas / 2
    alphas = (alphas + eps) * (1 - eps) / (1 + eps)
    return alphas


def ddpm_schedule(num_step: int) -> tuple[jax.Array, jax.Array]:
    """Generate DDPM schedule."""
    eps = 1e-4
    beta = ((num_step**2) * jnp.log(1 / (1 - eps)) + jnp.log(eps)) / (num_step - 1)
    gamma = (
        -num_step * (num_step * jnp.log(1 / (1 - eps)) + jnp.log(eps)) / (num_step - 1)
    )
    t = jnp.linspace(1.0 / num_step, 1.0, num_step)
    alphas = jnp.exp(-beta * t - gamma * t**2)
    return alphas


def fitness_mapping_energy(fitness: Fitness, temperature: float = 1.0) -> jax.Array:
    """Transform fitness into probabilities with softmax."""
    fitness = -fitness / temperature
    fitness = fitness - fitness.max() + 5.0
    return jnp.exp(fitness)


class DiffusionEvolution(PopulationBasedAlgorithm):
    """Diffusion Evolution."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        num_generations: int = 128,
        fitness_mapping: Callable = fitness_mapping_energy,
        alpha_schedule: Callable = cosine_schedule,
        num_latent_dims: int | None = None,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize Diffusion Evolution."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.num_generations = num_generations
        self.num_latent_dims = num_latent_dims
        self.fitness_mapping = fitness_mapping
        self.alpha_schedule = alpha_schedule

    @property
    def _default_params(self) -> Params:
        alphas = self.alpha_schedule(num_step=self.num_generations)
        alphas = alphas[::-1]  # reverse alpha_t (from t=T to t=0)

        return Params(
            std_m=1.0,
            scale_factor=1.0,
            alphas=alphas,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        # Generate projection matrix if num_latent_dims provided
        if self.num_latent_dims is not None:
            latent_projection = jax.random.normal(
                key, (self.num_dims, self.num_latent_dims)
            ) / jnp.sqrt(self.num_dims)  # Not in line with Linderstrauss lemma
        else:
            latent_projection = jnp.eye(self.num_dims)

        state = State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            std=params.std_m,
            latent_projection=latent_projection,
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        fitness = self.fitness_mapping(state.fitness)

        # Project population to latent space
        population = state.population / params.scale_factor
        population_latent = population @ state.latent_projection

        # Get alpha_t and alpha_{t-1}
        alpha_t = params.alphas[state.generation_counter]
        alpha_tm1 = params.alphas[state.generation_counter + 1]

        # Estimate the original point x_0
        x_0_hat = estimate_x_0(
            population,
            population_latent,
            fitness,
            alpha_t,
        )

        # Compute x_{t-1}
        population = ddim_step(
            key,
            population,
            x_0_hat,
            alpha_t,
            alpha_tm1,
            state.std,
        )

        # Scale population
        population = params.scale_factor * population
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Update population and fitness
        return state.replace(population=population, fitness=fitness)


def estimate_x_0(
    population: Population,
    population_latent: Population,
    fitness: Fitness,
    alpha: jax.Array,
) -> jax.Array:
    """Estimate the initial point x_0."""

    def estimate(x_t):
        mu = jnp.sqrt(alpha) * population_latent
        std = jnp.sqrt(1 - alpha)
        prob = fitness * gaussian_prob(x_t, mu, std)

        x_0_hat = jnp.dot(prob, population) / jnp.clip(jnp.sum(prob), min=1e-08)
        return x_0_hat

    x_0_hat = jax.vmap(estimate)(population_latent)
    return x_0_hat


def gaussian_prob(x, mean, std):
    """Multivariate normal probability distribution function without normalization."""
    dist = jnp.linalg.norm(x - mean, axis=-1)
    return jnp.exp(-0.5 * (dist / std) ** 2)


def ddim_step(key, x_t, x_0, alpha_t, alpha_tm1, std_m):
    """One step of the DDIM algorithm."""
    std_t = ddim_std(std_m, alpha_t, alpha_tm1)
    eps = (x_t - jnp.sqrt(alpha_t) * x_0) / jnp.sqrt(1.0 - alpha_t)
    x_tm1 = (
        jnp.sqrt(alpha_tm1) * x_0
        + jnp.sqrt(1 - alpha_tm1 - std_t**2) * eps
        + std_t * jax.random.normal(key, x_0.shape)
    )
    return x_tm1


def ddim_std(std_m: float, alpha_t: float, alpha_tm1: float) -> float:
    """Compute the default std for the DDIM algorithm."""
    return std_m * jnp.sqrt((1 - alpha_tm1) / (1 - alpha_t) * (1 - alpha_t / alpha_tm1))
