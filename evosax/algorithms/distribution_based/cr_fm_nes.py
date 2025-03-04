"""Cost-Reduced Fast-Moving Natural Evolution Strategy (Nomura & Ono, 2022).

Reference: https://arxiv.org/abs/2201.11422
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    p_std: jax.Array
    p_c: jax.Array
    D: jax.Array
    v: jax.Array
    z: jax.Array
    y: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    weights_hat: jax.Array
    weights: jax.Array
    mu_eff: float
    c_std: float
    c_c: float
    c1: float
    chi_n: float
    h_inv: float
    alpha_dist: float
    lr_mean: float = 1.0
    lr_move_std: float = 0.1
    lr_stag_std: float = 0.1
    lr_conv_std: float = 0.1
    lr_B: float = 0.1


def get_h_inv(dim: int) -> float:
    import math

    dim = min(dim, 2000)
    f = lambda a: ((1.0 + a * a) * math.exp(a * a / 2.0) / 0.24) - 10.0 - dim
    f_prime = lambda a: (1.0 / 0.24) * a * math.exp(a * a / 2.0) * (3.0 + a * a)
    h_inv = 1.0
    counter = 0
    while abs(f(h_inv)) > 1e-10:
        counter += 1
        h_inv = h_inv - 0.5 * (f(h_inv) / f_prime(h_inv))
    return h_inv


class CR_FM_NES(DistributionBasedAlgorithm):
    """Cost-Reduced Fast-Moving Natural Evolution Strategy (CR-FM-NES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize CR-FM-NES."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        self.elite_ratio = 0.5

    @property
    def _default_params(self) -> Params:
        weights_hat = jnp.log(self.population_size / 2 + 1) - jnp.log(
            jnp.arange(1, self.population_size + 1)
        )
        weights_hat = weights_hat * (weights_hat >= 0)
        weights = weights_hat / sum(weights_hat) - 1 / self.population_size

        mueff = 1 / (
            (weights + (1 / self.population_size)).T
            @ (weights + (1 / self.population_size))
        )
        c_std = (mueff + 2.0) / (self.num_dims + mueff + 5.0)
        c_c = (4.0 + mueff / self.num_dims) / (
            self.num_dims + 4.0 + 2.0 * mueff / self.num_dims
        )
        c1_cma = 2.0 / (jnp.power(self.num_dims + 1.3, 2) + mueff)
        chi_n = jnp.sqrt(self.num_dims) * (
            1.0
            - 1.0 / (4.0 * self.num_dims)
            + 1.0 / (21.0 * self.num_dims * self.num_dims)
        )

        # CR-FM-NES specific parameters
        h_inv = get_h_inv(self.num_dims)
        alpha_dist = h_inv * jnp.minimum(
            1.0, jnp.sqrt(self.population_size / self.num_dims)
        )
        lr_move_std = 1.0
        lr_stag_std = jnp.tanh(
            (0.024 * self.population_size + 0.7 * self.num_dims + 20.0)
            / (self.num_dims + 12.0)
        )
        lr_conv_std = 2.0 * jnp.tanh(
            (0.025 * self.population_size + 0.75 * self.num_dims + 10.0)
            / (self.num_dims + 4.0)
        )
        c1 = c1_cma * (self.num_dims - 5) / 6
        lr_B = jnp.tanh(
            (jnp.minimum(0.02 * self.population_size, 3 * jnp.log(self.num_dims)) + 5)
            / (0.23 * self.num_dims + 25)
        )

        params = Params(
            std_init=1.0,
            weights=weights,
            weights_hat=weights_hat,
            mu_eff=mueff,
            c_std=c_std,
            c_c=c_c,
            c1=c1,
            chi_n=chi_n,
            h_inv=h_inv,
            alpha_dist=alpha_dist,
            lr_move_std=lr_move_std,
            lr_stag_std=lr_stag_std,
            lr_conv_std=lr_conv_std,
            lr_B=lr_B,
        )
        return params

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_std=jnp.zeros((self.num_dims,)),
            p_c=jnp.zeros((self.num_dims,)),
            v=jax.random.normal(key, shape=(self.num_dims,)) / jnp.sqrt(self.num_dims),
            D=jnp.ones((self.num_dims,)),
            z=jnp.zeros((self.population_size, self.num_dims)),
            y=jnp.zeros((self.population_size, self.num_dims)),
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
        z = jax.random.normal(key, (self.population_size // 2, self.num_dims))
        z = jnp.concatenate([z, -z])

        norm_v_sq = jnp.sum(jnp.square(state.v))
        v_bar = state.v / jnp.sqrt(norm_v_sq)

        y = z + (jnp.sqrt(1 + norm_v_sq) - 1) * (z @ v_bar)[:, None] * v_bar
        x = state.mean + state.std * state.D * y
        return x, state.replace(z=z, y=y)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Step 3: Sort population by fitness
        idx = fitness.argsort()
        x = population[idx]
        z = state.z[idx]
        y = state.y[idx]

        # Step 4: Update evolution path p_std
        p_std = (1 - params.c_std) * state.p_std + jnp.sqrt(
            params.c_std * (2 - params.c_std) * params.mu_eff
        ) * jnp.dot(params.weights, z)
        norm_p_std = jnp.linalg.norm(p_std)

        # Step 6: Set the weights
        weights_dist_hat = jnp.exp(params.alpha_dist * jnp.linalg.norm(z, axis=-1))
        weights_dist = params.weights_hat * weights_dist_hat
        weights_dist = weights_dist / jnp.sum(weights_dist) - 1.0 / self.population_size

        movement_cond = params.chi_n <= norm_p_std
        weights = jnp.where(movement_cond, weights_dist, params.weights)

        # Step 7: Set the learning rate
        lr_std = jnp.select(
            condlist=[movement_cond, 0.1 * params.chi_n <= norm_p_std],
            choicelist=[params.lr_move_std, params.lr_stag_std],
            default=params.lr_conv_std,
        )

        # Update evolution path p_c and mean
        wxm = jnp.dot(weights, x - state.mean)
        p_c = (1.0 - params.c_c) * state.p_c + jnp.sqrt(
            params.c_c * (2 - params.c_c) * params.mu_eff
        ) * wxm / state.std
        mean = state.mean + params.lr_mean * wxm

        norm_v_2 = jnp.sum(jnp.square(state.v))
        norm_v_4 = norm_v_2**2
        norm_v = jnp.sqrt(norm_v_2)
        v_bar = state.v / norm_v

        exY = jnp.append(y, p_c[None, :] / state.D, axis=0)
        yy = exY * exY
        ip_yvbar = exY @ v_bar
        y_v_bar = exY * v_bar
        gamma_v = 1.0 + norm_v_2
        v_barbar = v_bar * v_bar
        alpha_vd = jnp.minimum(
            1,
            jnp.sqrt(norm_v_4 + (2 * gamma_v - jnp.sqrt(gamma_v)) / jnp.max(v_barbar))
            / (2 + norm_v_2),
        )
        t = exY * ip_yvbar[:, None] - 0.5 * v_bar * (ip_yvbar**2 + gamma_v)[:, None]
        b = -(1 - alpha_vd**2) * norm_v_4 / gamma_v + 2 * alpha_vd**2
        H = 2 * jnp.ones((self.num_dims,)) - (b + 2 * alpha_vd**2) * v_barbar
        invH = 1 / H
        s_step1 = (
            yy
            - norm_v_2 / gamma_v * (y_v_bar * ip_yvbar[:, None])
            - jnp.ones((self.population_size + 1, self.num_dims))
        )
        ip_vbart = jnp.dot(t, v_bar)
        s_step2 = s_step1 - alpha_vd / gamma_v * (
            (2 + norm_v_2) * (t * v_bar) - norm_v_2 * jnp.outer(ip_vbart, v_barbar)
        )
        invHvbarbar = invH * v_barbar
        ip_s_step2invHvbarbar = jnp.dot(s_step2, invHvbarbar)
        s = (s_step2 * invH) - b / (1 + b * jnp.dot(v_barbar, invHvbarbar)) * jnp.outer(
            ip_s_step2invHvbarbar, invHvbarbar
        )
        ip_svbarbar = jnp.dot(s, v_barbar)
        t = t - alpha_vd * (
            (2 + norm_v_2) * (s * v_bar) - jnp.outer(ip_svbarbar, v_bar)
        )

        # Update v, D covariance ingredients
        exw = jnp.concatenate([params.lr_B * weights, jnp.array([params.c1])])
        v = state.v + jnp.dot(exw, t) / norm_v
        D = state.D + jnp.dot(exw, s) * state.D

        # Calculate detA
        nthrootdetA = jnp.exp(
            jnp.sum(jnp.log(D)) / self.num_dims
            + jnp.log(1 + jnp.dot(v, v)) / (2 * self.num_dims)
        )
        D = D / nthrootdetA

        # Update std
        G_s = (
            jnp.sum(
                jnp.dot(
                    weights, z * z - jnp.ones((self.population_size, self.num_dims))
                )
            )
            / self.num_dims
        )
        std = state.std * jnp.exp(lr_std / 2 * G_s)
        return state.replace(mean=mean, std=std, p_std=p_std, p_c=p_c, v=v, D=D)
