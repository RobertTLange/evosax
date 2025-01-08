from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..strategy import Strategy
from ..utils import get_best_fitness_member


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: chex.Array
    archive: chex.Array
    x0_est: chex.Array
    alphas: chex.Array
    alphas_past: chex.Array
    latent_projection: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    sigma_init: float = 1.0
    init_scale: float = 1.0
    fitness_map_temp: float = 1.0
    fitness_map_power: float = 1.0
    fitness_map_l2_factor: float = 0.0
    scale_factor: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class DiffusionEvolution(Strategy):

    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        num_generations: int = 100,
        fitness_mapping: str = "energy",
        alpha_schedule: str = "cosine",
        sigma_init: float = 0.03,
        scale_factor: float = 1.0,
        init_scale: float = 1.0,
        num_latent_dims: Optional[int] = None,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float],
    ):
        """Diffusion Evolution (Zhang et al., 2024)
        Reference: https://arxiv.org/pdf/2410.02543"""
        super().__init__(
            popsize, num_dims, pholder_params, n_devices=n_devices, **fitness_kwargs
        )
        self.strategy_name = "DiffusionEvolution"
        self.sigma_init = sigma_init
        self.scale_factor = scale_factor
        self.init_scale = init_scale
        self.num_generations = num_generations
        self.num_latent_dims = num_latent_dims

        if alpha_schedule not in ["cosine", "ddpm", "ddim"]:
            raise ValueError(
                f"alpha_schedule must be one of 'cosine', 'ddpm', 'ddim', got {alpha_schedule}"
            )
        else:
            self.alpha_map = alpha_schedule_dict[alpha_schedule]

        if fitness_mapping not in ["identity", "energy", "power"]:
            raise ValueError(
                f"fitness_mapping must be one of 'identity', 'energy', 'power', got {fitness_mapping}"
            )
        else:
            self.fitness_map = fitness_mapping_dict[fitness_mapping]

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(
            sigma_init=self.sigma_init,
            scale_factor=self.scale_factor,
            init_scale=self.init_scale,
        )

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = (
            jax.random.normal(
                rng,
                (self.popsize, self.num_dims),
            )
            * params.init_scale
        )
        alphas_now, alphas_past = self.alpha_map(num_step=self.num_generations)

        # Generate projection matrix if num_latent_dims provided
        if self.num_latent_dims is not None:
            latent_projection = jax.random.normal(
                rng, (self.num_dims, self.num_latent_dims)
            ) / (self.num_dims**0.5)
        else:
            latent_projection = jnp.eye(self.num_dims)

        state = EvoState(
            mean=initialization.mean(axis=0),
            sigma=params.sigma_init,
            x0_est=initialization,
            archive=initialization,  # last evaluations
            alphas=alphas_now,
            alphas_past=alphas_past,
            latent_projection=latent_projection,
            best_member=initialization.mean(axis=0),
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """
        `ask` for new proposed candidates to evaluate next.
        """
        alpha_t = state.alphas[state.gen_counter - 1]
        alpha_pt = state.alphas_past[state.gen_counter - 1]
        # only do ddim step if not at 1st gen otw return init pop
        x_next = jax.lax.select(
            state.gen_counter == 0,
            state.archive,
            ddim_step(
                rng,
                state.archive,
                state.x0_est,
                alpha_t,
                alpha_pt,
                state.sigma,
            ),
        )
        x_scaled = x_next * params.scale_factor
        return x_scaled, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """
        `tell` update to ES state.
        """
        # map fitness to density (integrates to 1)
        # note -> by default reference implementation maximizes!
        # evosax, on the other hand, minimizes by default
        fitness_mapped = self.fitness_map(
            fitness,  # maximize by default!!!
            params.fitness_map_temp,
            params.fitness_map_power,
            params.fitness_map_l2_factor,
        )
        # rescale x to undo scaling in ask
        x_rescale = x / params.scale_factor
        x_latent = x_rescale @ state.latent_projection
        alpha_t = state.alphas[state.gen_counter]
        x0_est = estimate_x0(
            x_rescale,
            x_latent,
            fitness_mapped,
            alpha_t,
        )
        best_member, _ = get_best_fitness_member(x, fitness, state, False)
        # print("x0_est", x0_est[:2])
        return state.replace(
            mean=best_member,
            archive=x_rescale,
            x0_est=x0_est,
        )


def map_fitness2identity(
    fitness: chex.Array, temp: float, power: float, l2_factor: float
) -> chex.Array:
    """Map fitness to density."""
    l2_norm = jnp.linalg.norm(fitness) ** 2
    identity_out = fitness * jnp.exp(-1.0 * l2_norm * l2_factor)
    return identity_out


def map_fitness2energy(
    fitness: chex.Array, temp: float, power: float, l2_factor: float
) -> chex.Array:
    """Map fitness to energy."""
    power_temp = -fitness / temp
    power_temp = power_temp - power_temp.max() + 5
    energy_out = jnp.exp(power_temp)
    return energy_out


def map_fitness2power(
    fitness: chex.Array, temp: float, power: float, l2_factor: float
) -> chex.Array:
    """Map fitness to power."""
    return jnp.power(fitness / temp, power)


fitness_mapping_dict = {
    "identity": map_fitness2identity,
    "energy": map_fitness2energy,
    "power": map_fitness2power,
}


def estimate_x0(
    x: chex.Array,
    x_latent: chex.Array,
    fitness: chex.Array,
    alpha: chex.Array,
) -> chex.Array:
    """Estimate the initial point."""
    # Uniform density over pop. - normalization => cancels out
    p_x_t = jnp.ones(x.shape[0]) / x.shape[0]

    # Estimate the original point
    def estimate(x_t, p_x_t):
        mu = x_latent * alpha**0.5
        sigma = (1 - alpha) ** 0.5
        p_diffusion = single_point_gaussian_prob(x_t, mu, sigma)
        prob = (fitness + 1e-09) * (p_diffusion + 1e-09) / (p_x_t + 1e-09)
        # Compute normalization term
        z = jnp.sum(prob)
        x0_est = jnp.sum(jnp.expand_dims(prob, axis=1) * x, axis=0) / (z + 1e-09)
        return x0_est

    x0_est = jax.vmap(estimate, in_axes=(0, 0))(x_latent, p_x_t)
    return x0_est


def single_point_gaussian_prob(x_i, mu, sigma):
    """Compute the probability under multiple Gaussian means."""
    dist = jnp.linalg.norm(x_i - mu, axis=-1)
    return jnp.exp(-(dist**2) / (2 * sigma**2))


def generate_cosine_schedule(num_step: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the cosine schedule upfront."""
    alphas = jnp.cos(jnp.linspace(0, jnp.pi, num_step)) + 1
    alphas = alphas / 2
    # Reverse and shift alphas to get alpha_past
    alphas = alphas[::-1]
    alphas_now = alphas[:-1]
    alphas_past = alphas[1:]
    return alphas_now, alphas_past


def generate_ddpm_schedule(num_step: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the DDPM schedule upfront."""
    eps = 1e-4
    beta = ((num_step**2) * jnp.log(1 / (1 - eps)) + jnp.log(eps)) / (num_step - 1)
    gamma = (
        -num_step * (num_step * jnp.log(1 / (1 - eps)) + jnp.log(eps)) / (num_step - 1)
    )
    t = jnp.linspace(1.0 / num_step, 1.0, num_step)
    alphas = jnp.exp(-beta * t - gamma * t**2)
    # Reverse and shift alphas to get alpha_past
    alphas = alphas[::-1]
    alphas_now = alphas[:-1]
    alphas_past = alphas[1:]
    return alphas_now, alphas_past


def generate_ddim_schedule(
    num_step: int, power: float = 1.0, eps: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the DDIM schedule upfront."""
    # Compute alpha values based on DDIM schedule
    alphas = (
        jnp.linspace(
            1 - eps,
            (eps * eps) ** (1 / power),
            num_step,
        )
        ** power
    )
    # Reverse and shift alphas to get alpha_past
    alphas = alphas[::-1]
    alphas_now = alphas[:-1]
    alphas_past = alphas[1:]
    return alphas_now, alphas_past


alpha_schedule_dict = {
    "cosine": generate_cosine_schedule,
    "ddpm": generate_ddpm_schedule,
    "ddim": generate_ddim_schedule,
}


def ddim_step(rng, xt, x0, alpha_t, alpha_tp, noise_scale):
    """One step of the DDIM algorithm."""
    sigma = ddpm_sigma(alpha_t, alpha_tp) * noise_scale
    # print("Sigma", sigma, alpha_t, alpha_tp, noise_scale)
    eps = (xt - (alpha_t**0.5) * x0) / (1.0 - alpha_t) ** 0.5
    x_next = (
        (alpha_tp**0.5) * x0
        + ((1 - alpha_tp - sigma**2) ** 0.5) * eps
        + sigma * jax.random.normal(rng, x0.shape)
    )
    return x_next


def ddpm_sigma(alpha_t, alpha_tp):
    """Compute the default sigma for the DDPM algorithm."""
    return ((1 - alpha_tp) / (1 - alpha_t) * (1 - alpha_t / alpha_tp)) ** 0.5
