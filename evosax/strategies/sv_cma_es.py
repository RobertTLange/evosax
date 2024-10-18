from typing import Optional

import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, PRNGKey
from flax.struct import dataclass

from evosax.strategies.cma_es import get_cma_elite_weights, update_p_c, update_p_sigma, sample, update_sigma, update_covariance, EvoParams, CMA_ES
from evosax.utils.eigen_decomp import full_eigen_decomp


class Kernel:
    def __call__(self, x1: Array, x2: Array, bandwidth: float) -> Array:
        pass

class RBF(Kernel):
    def __call__(self, x1: Array, x2: Array, bandwidth: float) -> Array:
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / bandwidth)


@dataclass
class EvoState:
    p_sigma: Array
    p_c: Array
    C: Array
    D: Optional[Array]
    B: Optional[Array]
    mean: Array
    sigma: Array
    weights: Array
    weights_truncated: Array
    best_member: Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0
    bandwidth: float = 1.
    alpha: float = 1.


class SV_CMA_ES(CMA_ES):
    def __init__(
        self,
        npop: int,
        subpopsize: int,
        kernel: Kernel,
        num_dims: Optional[int] = None,
        pholder_params: Optional[ArrayTree | Array] = None,
        elite_ratio: float = 0.5,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: bool | int | float
    ):
        """Stein Variational CMA-ES (Braun et al., 2024)
        Reference: https://arxiv.org/abs/2410.10390"""
        self.npop = npop
        self.subpopsize = subpopsize
        popsize = int(npop * subpopsize)
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            elite_ratio,
            sigma_init,
            mean_decay,
            n_devices,
            **fitness_kwargs
        )
        self.elite_popsize = max(1, int(self.subpopsize * self.elite_ratio))
        self.strategy_name = "SV_CMA_ES"
        self.kernel = kernel

    def initialize_strategy(
        self, rng: PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        weights, weights_truncated, _, _, _ = get_cma_elite_weights(
            self.subpopsize, self.elite_popsize, self.num_dims, self.max_dims_sq
        )
        # Initialize evolution paths & covariance matrix
        initialization = jax.random.uniform(
            rng,
            (self.npop, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )

        state = EvoState(
            p_sigma=jnp.zeros((self.npop, self.num_dims)),
            p_c=jnp.zeros((self.npop, self.num_dims)),
            sigma=jnp.ones(self.npop) * params.sigma_init,
            mean=initialization,
            C=jnp.tile(jnp.eye(self.num_dims), (self.npop, 1, 1)),
            D=None,
            B=None,
            weights=weights,
            weights_truncated=weights_truncated,
            best_member=initialization[0],  # Take any random member of the means
        )
        return state

    def ask_strategy(
        self, rng: PRNGKey, state: EvoState, params: EvoParams
    ) -> [Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        Cs, Bs, Ds = jax.vmap(full_eigen_decomp, (0, 0, 0, None))(
            state.C, state.B, state.D, state.gen_counter
        )
        keys = jax.random.split(rng, num=self.npop)
        x = jax.vmap(sample, (0, 0, 0, 0, 0, None, None))(
            keys,
            state.mean,
            state.sigma,
            Bs,
            Ds,
            self.num_dims,
            self.subpopsize,
        )

        # Reshape for evaluation
        x = x.reshape(self.popsize, self.num_dims)

        return x, state.replace(C=Cs, B=Bs, D=Ds)

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

        # Compute grads
        y_ks, y_ws = jax.vmap(cmaes_grad, (0, 0, 0, 0, None))(
            x,
            fitness,
            state.mean,
            state.sigma,
            state.weights_truncated
        )

        # Compute kernel grads
        bandwidth = state.bandwidth
        kernel_grads = jax.vmap(
            lambda xi: jnp.mean(
                jax.vmap(lambda xj: jax.grad(self.kernel)(xj, xi, bandwidth))(state.mean),
                axis=0
            )
        )(state.mean)

        # Update means using the kernel gradients
        alpha = state.alpha
        projected_steps = y_ws + alpha * kernel_grads / state.sigma[:, None]
        means = state.mean + params.c_m * state.sigma[:, None] * projected_steps

        # Search distribution updates
        p_sigmas, C_2s, Cs, Bs, Ds = jax.vmap(update_p_sigma, (0, 0, 0, 0, 0, None, None, None))(
            state.C,
            state.B,
            state.D,
            state.p_sigma,
            projected_steps,
            params.c_sigma,
            params.mu_eff,
            state.gen_counter,
        )

        p_cs, norms_p_sigma, h_sigmas = jax.vmap(update_p_c, (0, 0, 0, None, 0, None, None, None, None))(
            means,
            p_sigmas,
            state.p_c,
            state.gen_counter + 1,
            projected_steps,
            params.c_sigma,
            params.chi_n,
            params.c_c,
            params.mu_eff,
        )

        Cs = jax.vmap(update_covariance, (0, 0, 0, 0, 0, 0, None, None, None, None))(
            means,
            p_cs,
            Cs,
            y_ks,
            h_sigmas,
            C_2s,
            state.weights,
            params.c_c,
            params.c_1,
            params.c_mu
        )

        sigmas = jax.vmap(update_sigma, (0, 0, None, None, None))(
            state.sigma,
            norms_p_sigma,
            params.c_sigma,
            params.d_sigma,
            params.chi_n,
        )

        return state.replace(
            mean=means, p_sigma=p_sigmas, C=Cs, B=Bs, D=Ds, p_c=p_cs, sigma=sigmas
        )


def cmaes_grad(
    x: Array,
    fitness: Array,
    mean: Array,
    sigma: float,
    weights_truncated: Array,
) -> [Array, Array]:
    """Approximate gradient using samples from a search distribution."""
    # get sorted solutions
    concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
    sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
    # get the scores
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = (x_k - mean) / sigma  # ~ N(0, C)
    grad = jnp.dot(weights_truncated.T, y_k)  # y_w can be seen as score estimate of CMA-ES

    return y_k, grad


if __name__ == "__main__":
    from typing import Optional

    import numpy as np
    import matplotlib.pyplot as plt


    def KDE(kernel: Kernel, modes: Array, weights: Optional[Array] = None, bandwidth: float = 1.):
        """Kernel density estimation."""
        if weights is None:
            weights = jnp.ones(modes.shape[0]) / modes.shape[0]
        return lambda xi: jnp.sum(
            jax.vmap(kernel, in_axes=(None, 0, None))(xi, modes, bandwidth) * weights
        )


    def plot_pdf(pdf, ax: Optional = None, xmin: float = -3., xmax: float = 3.):
        x = np.linspace(xmin, xmax, 50)
        x_ = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
        energies = pdf(x_)

        plt.figure(figsize=(7, 7))
        if ax is not None:
            ax.contourf(x, x, energies.reshape(50, 50), levels=20)  # , cmap="Greys"
        else:
            plt.contourf(x, x, energies.reshape(50, 50), levels=20)  # , cmap="Greys"


    def plot_particles_pdf(x, objective, score_fn, npop: int, xmin: float = -3., xmax: float = 3.):
        plot_pdf(objective, score_fn, xmin, xmax)
        plt.scatter(*x.T, color="salmon")
        # for xi in x.reshape(npop, -1, 2):
        #     plt.scatter(*xi.T)
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, xmax)
        # plt.show()


    class Benchmark:
        def __init__(self, lb: Array | float, ub: Array | float, dim: int, fglob: float, name: str) -> None:
            self.lower_bounds = lb
            self.upper_bounds = ub
            self.dim = dim
            self.fglob = fglob
            self.name = name

        def get_objective_derivative(self):
            pass

        def get_objective(self):
            return self.get_objective_derivative()[0]

        def plot(self, x: Array, lb: Array, ub: Array):
            """Plotting code. Works for synthetic benchmarks."""
            plot_particles_pdf(
                x[0],
                lambda y: jnp.exp(-self.get_objective()(y)),
                None,
                1,
                lb,
                ub
            )


    class GMM(Benchmark):
        def __init__(
            self,
            rng: PRNGKey,
            lb: float = -6.,
            ub: float = 6.,
            kernel_rad: float = 1.,
            n_modes: int = 4,
            dim: int = 2,
            name: str = "GMM"
        ) -> None:
            fglob = -1 / (kernel_rad * (2 ** dim + 1))  # pdf = height * width * npeaks != 1 solves to this
            super().__init__(lb, ub, dim, fglob, name)
            self.kernel_rad = kernel_rad

            # Instantiate problem
            rng_w, rng_m = jax.random.split(rng)
            self.weights = jax.random.uniform(rng_w, (n_modes,), minval=0., maxval=10.)
            self.weights /= jnp.sum(self.weights)
            self.modes = jax.random.uniform(rng_m, (n_modes, dim), minval=lb + 2,
                                            maxval=ub - 2)  # Add some slack beyond the bounds so they do not all overlap

        def get_objective_derivative(self):
            """Return the objective and its derivative functions."""
            eval_fn = jax.jit(lambda x: -jnp.log(KDE(RBF(), self.modes, self.weights, self.kernel_rad)(x)))
            return jax.vmap(eval_fn), jax.vmap(jax.grad(lambda x: -eval_fn(x)))

    # Benchmark
    rng = jax.random.PRNGKey(2)
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    dim = 2
    bench = GMM(init_rng, dim=dim, n_modes=4, lb=-6., ub=6.)
    n_iter = 1_000
    npop = 100
    popsize = 4
    cb_freq = 50

    def plot_cb(x):
        bench.plot((x, None), bench.lower_bounds - 2, bench.upper_bounds + 2)
        plt.show()


    rng, rng_init, rng_sample = jax.random.split(rng, 3)
    strategy = SV_CMA_ES(npop=npop, subpopsize=popsize, kernel=RBF(), num_dims=bench.dim, elite_ratio=0.5, sigma_init=.05)
    es_params = strategy.default_params.replace(
        init_min=bench.lower_bounds,
        init_max=bench.upper_bounds,
        clip_min=bench.lower_bounds - 2,
        clip_max=bench.upper_bounds + 2
    )
    state = strategy.initialize(rng_init, es_params)
    state = state.replace(alpha=1., bandwidth=.5)

    # Get objective
    objective_fn, score_fn = bench.get_objective_derivative()

    samples = []
    for t in range(n_iter):
        rng, rng_gen = jax.random.split(rng)
        x, state = strategy.ask(rng_gen, state, es_params)
        fitness = objective_fn(x)  # Evaluate score for gradient-based SVGD
        state = strategy.tell(x, fitness, state, es_params)

        if t % cb_freq == 0:
            print(t + 1, fitness.min())
            if plot_cb:
                plot_cb(state.mean)
