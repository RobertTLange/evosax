from typing import Optional, Type, Union

import jax
from flax import struct
from chex import Array, ArrayTree
import jax.numpy as jnp

from evosax.core import OptState, exp_decay
from evosax.strategies.open_es import EvoParams, OpenES
from evosax.utils.kernel import RBF, Kernel


@struct.dataclass
class EvoState:
    mean: Array
    sigma: Array
    opt_state: OptState
    best_member: Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0
    bandwidth: float = 1.
    alpha: float = 1.


class SV_OpenES(OpenES):
    def __init__(
        self,
        npop: int,
        subpopsize: int,
        kernel: Type[Kernel] = RBF,
        num_dims: Optional[int] = None,
        pholder_params: Optional[ArrayTree | Array] = None,
        use_antithetic_sampling: bool = True,
        opt_name: str = "adam",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        sigma_init: float = 0.03,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Stein Variational OpenAI-ES (Liu et al., 2017)
        Reference: https://arxiv.org/abs/1704.02399"""
        super().__init__(
            npop * subpopsize,
            num_dims,
            pholder_params,
            use_antithetic_sampling,
            opt_name,
            lrate_init,
            lrate_decay,
            lrate_limit,
            sigma_init,
            sigma_decay,
            sigma_limit,
            mean_decay,
            n_devices,
            **fitness_kwargs
        )
        assert not subpopsize & 1, "Sub-population size size must be even"
        self.strategy_name = "SV_OpenAI_ES"
        self.npop = npop
        self.subpopsize = subpopsize
        self.kernel = kernel()

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        opt_params = self.optimizer.default_params.replace(
            lrate_init=self.lrate_init,
            lrate_decay=self.lrate_decay,
            lrate_limit=self.lrate_limit,
        )
        return EvoParams(
            opt_params=opt_params,
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
        )

    def initialize_strategy(
        self, rng: jax.random.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        x_init = jax.random.uniform(
            rng,
            (self.npop, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max
        )
        state = EvoState(
            mean=x_init,
            sigma=jnp.ones((self.npop, self.num_dims)) * params.sigma_init,
            opt_state=jax.vmap(lambda _: self.optimizer.initialize(params.opt_params))(jnp.arange(self.npop)),
            best_member=x_init[0],  # pholder best
        )

        return state

    def ask_strategy(
        self, rng: jax.random.PRNGKey, state: EvoState, params: EvoParams
    ) -> [Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        if self.use_antithetic_sampling:
            z_plus = jax.random.normal(
                rng,
                (self.npop, int(self.subpopsize / 2), self.num_dims),
            )
            z = jnp.concatenate([z_plus, -1.0 * z_plus], axis=1)
        else:
            z = jax.random.normal(rng, (self.npop, self.subpopsize, self.num_dims))

        x = state.mean[:, None] + state.sigma[:, None] * z
        x = x.reshape(self.popsize, self.num_dims)

        return x, state

    def tell_strategy(
        self,
        x: Array,
        fitness: Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        x = x.reshape(self.npop, self.subpopsize, self.num_dims)
        fitness = fitness.reshape(self.npop, self.subpopsize)

        # Compute MC gradients from fitness scores
        noise = (state.mean[:, None] - x) / state.sigma[:, None]
        scores = jnp.einsum("ijk,ij->ik", noise, fitness) / (self.subpopsize * state.sigma)

        # Compute SVGD steps
        svgd_scores = svgd_grad(state.mean, scores, self.kernel, state.bandwidth)
        svgd_kerns = svgd_kern(state.mean, scores, self.kernel, state.bandwidth)
        gradients = -(svgd_scores + svgd_kerns * state.alpha)  # flip the grads for minimization

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = jax.vmap(self.optimizer.step, (0, 0, 0, None))(
            state.mean, gradients, state.opt_state, params.opt_params
        )
        opt_state = jax.vmap(self.optimizer.update, (0, None))(opt_state, params.opt_params)
        sigma = jax.vmap(exp_decay, (0, None, None))(state.sigma, params.sigma_decay, params.sigma_limit)

        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state)


def svgd_kern(x: Array, scores: Array, kernel: Kernel, bandwidth: float) -> Array:
    """SVGD repulsive force."""
    phi = lambda xi: jnp.mean(
        jax.vmap(lambda xj, scorej: jax.grad(kernel)(xj, xi, bandwidth))(x, scores),
        axis=0
    )
    return jax.vmap(phi)(x)


def svgd_grad(x: Array, scores: Array, kernel: Kernel, bandwidth: float) -> Array:
    """SVGD driving force."""
    phi = lambda xi: jnp.mean(
        jax.vmap(lambda xj, scorej: kernel(xj, xi, bandwidth) * scorej)(x, scores),
        axis=0
    )
    return jax.vmap(phi)(x)
