from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
import math
from flax import struct
from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    v: chex.Array
    D: chex.Array
    p_sigma: chex.Array
    p_c: chex.Array
    w_rank_hat: chex.Array
    w_rank: chex.Array
    z: chex.Array
    y: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    mu_eff: float
    c_s: float
    c_c: float
    c1: float
    chi_N: float
    h_inv: float
    alpha_dist: float
    lrate_mean: float = 1.0
    lrate_move_sigma: float = 0.1
    lrate_stag_sigma: float = 0.1
    lrate_conv_sigma: float = 0.1
    lrate_B: float = 0.1
    sigma_init: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


def get_recombination_weights(popsize: int) -> Tuple[chex.Array, chex.Array]:
    """Get recombination weights for different ranks."""

    def get_weight(i):
        return jnp.log(popsize / 2 + 1) - jnp.log(i)

    w_rank_hat = jax.vmap(get_weight)(jnp.arange(1, popsize + 1))
    w_rank_hat = w_rank_hat * (w_rank_hat >= 0)
    w_rank = w_rank_hat / sum(w_rank_hat) - (1.0 / popsize)
    return w_rank_hat.reshape(-1, 1), w_rank.reshape(-1, 1)


def get_h_inv(dim: int) -> float:
    dim = min(dim, 2000)
    f = lambda a: ((1.0 + a * a) * math.exp(a * a / 2.0) / 0.24) - 10.0 - dim
    f_prime = lambda a: (1.0 / 0.24) * a * math.exp(a * a / 2.0) * (3.0 + a * a)
    h_inv = 1.0
    counter = 0
    while abs(f(h_inv)) > 1e-10:
        counter += 1
        h_inv = h_inv - 0.5 * (f(h_inv) / f_prime(h_inv))
    return h_inv


def w_dist_hat(alpha_dist: float, z: chex.Array) -> chex.Array:
    return jnp.exp(alpha_dist * jnp.linalg.norm(z))


class CR_FM_NES(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Cost-Reduced Fast-Moving Natural ES (Nomura & Ono, 2022)
        Reference: https://arxiv.org/abs/2201.11422
        """
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            mean_decay,
            n_devices,
            **fitness_kwargs
        )
        assert not self.popsize & 1, "Population size must be even"
        self.strategy_name = "CR_FM_NES"

        # Set core kwargs es_params (sigma)
        self.sigma_init = sigma_init

    @property
    def default_params(self) -> EvoParams:
        """Return default parameters of evolutionary strategy."""
        w_rank_hat, w_rank = get_recombination_weights(self.popsize)
        mueff = 1 / (
            (w_rank + (1 / self.popsize)).T @ (w_rank + (1 / self.popsize))
        )
        c_s = (mueff + 2.0) / (self.num_dims + mueff + 5.0)
        c_c = (4.0 + mueff / self.num_dims) / (
            self.num_dims + 4.0 + 2.0 * mueff / self.num_dims
        )
        c1_cma = 2.0 / (jnp.power(self.num_dims + 1.3, 2) + mueff)
        chi_N = jnp.sqrt(self.num_dims) * (
            1.0
            - 1.0 / (4.0 * self.num_dims)
            + 1.0 / (21.0 * self.num_dims * self.num_dims)
        )
        h_inv = get_h_inv(self.num_dims)
        alpha_dist = h_inv * jnp.minimum(
            1.0, jnp.sqrt(self.popsize / self.num_dims)
        )
        lrate_move_sigma = 1.0
        lrate_stag_sigma = jnp.tanh(
            (0.024 * self.popsize + 0.7 * self.num_dims + 20.0)
            / (self.num_dims + 12.0)
        )
        lrate_conv_sigma = 2.0 * jnp.tanh(
            (0.025 * self.popsize + 0.75 * self.num_dims + 10.0)
            / (self.num_dims + 4.0)
        )
        c1 = c1_cma * (self.num_dims - 5) / 6
        lrate_B = jnp.tanh(
            (jnp.minimum(0.02 * self.popsize, 3 * jnp.log(self.num_dims)) + 5)
            / (0.23 * self.num_dims + 25)
        )
        params = EvoParams(
            lrate_move_sigma=lrate_move_sigma,
            lrate_stag_sigma=lrate_stag_sigma,
            lrate_conv_sigma=lrate_conv_sigma,
            lrate_B=lrate_B,
            mu_eff=mueff,
            c_s=c_s,
            c_c=c_c,
            c1=c1,
            chi_N=chi_N,
            alpha_dist=alpha_dist,
            h_inv=h_inv,
            sigma_init=self.sigma_init,
        )
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolutionary strategy."""
        rng_init, rng_v = jax.random.split(rng)
        initialization = jax.random.uniform(
            rng_init,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        w_rank_hat, w_rank = get_recombination_weights(self.popsize)
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init,
            v=jax.random.normal(rng_v, shape=(self.num_dims, 1))
            / jnp.sqrt(self.num_dims),
            D=jnp.ones([self.num_dims, 1]),
            p_sigma=jnp.zeros((self.num_dims, 1)),
            p_c=jnp.zeros((self.num_dims, 1)),
            z=jnp.zeros((self.num_dims, self.popsize)),
            y=jnp.zeros((self.num_dims, self.popsize)),
            w_rank_hat=w_rank_hat.reshape(-1, 1),
            w_rank=w_rank,
            best_member=initialization,
        )

        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        z_plus = jax.random.normal(
            rng,
            (int(self.popsize / 2), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        z = jnp.swapaxes(z, 0, 1)
        normv = jnp.linalg.norm(state.v)
        normv2 = normv ** 2
        vbar = state.v / normv

        # Rescale/reparametrize noise
        y = z + (jnp.sqrt(1 + normv2) - 1) * vbar @ (vbar.T @ z)
        x = state.mean[:, None] + state.sigma * y * state.D
        x = jnp.swapaxes(x, 0, 1)
        return x, state.replace(z=z, y=y)

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        ranks = fitness.argsort()
        z = state.z[:, ranks]
        y = state.y[:, ranks]
        x = jnp.swapaxes(x, 0, 1)[:, ranks]

        # Update evolution path p_sigma
        p_sigma = (1 - params.c_s) * state.p_sigma + jnp.sqrt(
            params.c_s * (2.0 - params.c_s) * params.mu_eff
        ) * (z @ state.w_rank)
        p_sigma_norm = jnp.linalg.norm(p_sigma)

        # Calculate distance weight
        w_tmp = state.w_rank_hat * jax.vmap(w_dist_hat, in_axes=(None, 1))(
            params.alpha_dist, z
        ).reshape(-1, 1)
        weights_dist = w_tmp / sum(w_tmp) - 1.0 / self.popsize

        # switching weights and learning rate
        p_sigma_cond = p_sigma_norm >= params.chi_N
        weights = jax.lax.select(p_sigma_cond, weights_dist, state.w_rank)
        lrate_sigma = jax.lax.select(
            p_sigma_cond, params.lrate_move_sigma, params.lrate_stag_sigma
        )
        lrate_sigma = jax.lax.select(
            p_sigma_norm >= 0.1 * params.chi_N,
            lrate_sigma,
            params.lrate_conv_sigma,
        )

        # update evolution path p_c and mean
        wxm = (x - state.mean[:, None]) @ weights
        p_c = (1.0 - params.c_c) * state.p_c + jnp.sqrt(
            params.c_c * (2.0 - params.c_c) * params.mu_eff
        ) * wxm / state.sigma
        mean = state.mean + params.lrate_mean * wxm.squeeze()

        normv = jnp.linalg.norm(state.v)
        vbar = state.v / normv
        normv2 = normv ** 2
        normv4 = normv2 ** 2

        exY = jnp.append(y, p_c / state.D, axis=1)
        yy = exY * exY
        ip_yvbar = vbar.T @ exY
        yvbar = exY * vbar
        gammav = 1.0 + normv2
        vbarbar = vbar * vbar
        alphavd = jnp.minimum(
            1,
            jnp.sqrt(
                normv4 + (2 * gammav - jnp.sqrt(gammav)) / jnp.max(vbarbar)
            )
            / (2 + normv2),
        )
        t = exY * ip_yvbar - vbar * (ip_yvbar ** 2 + gammav) / 2
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2
        H = jnp.ones([self.num_dims, 1]) * 2 - (b + 2 * alphavd ** 2) * vbarbar
        invH = H ** (-1)
        s_step1 = (
            yy
            - normv2 / gammav * (yvbar * ip_yvbar)
            - jnp.ones([self.num_dims, self.popsize + 1])
        )
        ip_vbart = vbar.T @ t
        s_step2 = s_step1 - alphavd / gammav * (
            (2 + normv2) * (t * vbar) - normv2 * vbarbar @ ip_vbart
        )
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = invHvbarbar.T @ s_step2
        s = (s_step2 * invH) - b / (
            1 + b * vbarbar.T @ invHvbarbar
        ) * invHvbarbar @ ip_s_step2invHvbarbar
        ip_svbarbar = vbarbar.T @ s
        t = t - alphavd * ((2 + normv2) * (s * vbar) - vbar @ ip_svbarbar)

        # update v, D covariance ingredients
        exw = jnp.append(
            params.lrate_B * weights,
            jnp.array([params.c1]).reshape(1, 1),
            axis=0,
        )
        v = state.v + (t @ exw) / normv
        D = state.D + (s @ exw) * state.D
        # calculate detA
        nthrootdetA = jnp.exp(
            jnp.sum(jnp.log(D)) / self.num_dims
            + jnp.log(1 + v.T @ v) / (2 * self.num_dims)
        )[0][0]
        D = D / nthrootdetA
        # update sigma
        G_s = (
            jnp.sum((z * z - jnp.ones([self.num_dims, self.popsize])) @ weights)
            / self.num_dims
        )
        sigma = state.sigma * jnp.exp(lrate_sigma / 2 * G_s)
        return state.replace(
            p_sigma=p_sigma, mean=mean, p_c=p_c, v=v, D=D, sigma=sigma
        )
