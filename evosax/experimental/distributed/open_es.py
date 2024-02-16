"""Distributed version of OpenAI-ES. Supports z-scoring fitness trafo only."""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional, Union
from flax import struct
from evosax import Strategy
from evosax.core import GradientOptimizer, OptState, OptParams, exp_decay


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    opt_state: OptState
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    opt_params: OptParams
    sigma_init: float = 0.04
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    init_min: float = -2.0
    init_max: float = 2.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class OpenES(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        opt_name: str = "adam",
        lrate_init: float = 0.01,
        lrate_decay: float = 0.999,
        lrate_limit: float = 0.001,
        sigma_init: float = 0.04,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
    ):
        """Pmapped version of OpenAI-ES (Salimans et al. (2017)
        Samples directly on different devices and updates mean using pmean grad.
        Reference: https://arxiv.org/pdf/1703.03864.pdf
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(popsize, num_dims, pholder_params)
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup", "adan"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "DistributedOpenES"
        self.lrate_init = lrate_init
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices

        # Mean exponential decay coefficient m' = coeff * m
        # Implements form of weight decay regularization
        self.mean_decay = mean_decay
        self.use_mean_decay = mean_decay > 0.0

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        opt_params = self.optimizer.default_params.replace(
            lrate_init=self.lrate_init,
            lrate_decay=self.lrate_decay,
            lrate_limit=self.lrate_limit,
        )
        es_params = EvoParams(
            opt_params=opt_params,
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
        )
        if self.n_devices > 1:
            es_params = jax.tree_map(
                lambda x: jnp.array([x] * self.n_devices), es_params
            )
        else:
            es_params = es_params
        return es_params

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        if self.n_devices > 1:
            mean, sigma, opt_state = self.multi_init(rng, params)
        else:
            mean, sigma, opt_state = self.single_init(rng, params)

        state = EvoState(
            mean=mean,
            sigma=sigma,
            opt_state=opt_state,
            best_member=mean,
        )
        return state

    def multi_init(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> Tuple[chex.Array, chex.Array, OptState]:
        """`initialize` the evolution strategy on multiple devices (same)."""
        # Use rng tile to create same random sample across devices
        batch_rng = jnp.tile(rng, (self.n_devices, 1))
        mean, sigma, opt_state = jax.pmap(self.single_init)(batch_rng, params)
        return mean, sigma, opt_state

    def single_init(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> Tuple[chex.Array, chex.Array, OptState]:
        """`initialize` the evolution strategy on a single device."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        opt_state = self.optimizer.initialize(params.opt_params)
        mean = initialization
        sigma = params.sigma_init
        return mean, sigma, opt_state

    def ask(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        if self.n_devices > 1:
            x = self.multi_ask(rng, state.mean, state.sigma)
        else:
            x = self.single_ask(rng, state.mean, state.sigma)

        if self.use_param_reshaper:
            if self.n_devices > 1:
                x = self.param_reshaper.multi_reshape(x)
            else:
                x = self.param_reshaper.reshape(x)
        return x, state

    def multi_ask(
        self, rng: chex.PRNGKey, mean: chex.Array, sigma: chex.Array
    ) -> chex.Array:
        """Pmapped antithetic sampling of noise and reparametrization."""
        # Use rng split to create different random sample across devices
        batch_rng = jax.random.split(rng, self.n_devices)
        x = jax.pmap(self.single_ask)(batch_rng, mean, sigma)
        return x

    def single_ask(
        self, rng: chex.PRNGKey, mean: chex.Array, sigma: chex.Array
    ) -> chex.Array:
        """Antithetic sampling of noise and reparametrization."""
        z_plus = jax.random.normal(
            rng,
            (int(self.popsize / (2 * self.n_devices)), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = mean + sigma * z
        return x

    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        if self.use_param_reshaper:
            if self.n_devices > 1:
                x = self.param_reshaper.multi_flatten(x)
            else:
                x = self.param_reshaper.flatten(x)

        if self.n_devices > 1:
            mean, sigma, opt_state = self.multi_tell(x, fitness, state, params)
        else:
            mean, sigma, opt_state = self.single_tell(x, fitness, state, params)

        # Exponentially decay mean if coefficient < 1.0
        if self.use_mean_decay:
            mean = mean * (1 - self.mean_decay)

        # TODO(RobertTLange): Add tracking of best member/fitness score
        return state.replace(mean=mean, sigma=sigma, opt_state=opt_state)

    def multi_tell(
        self, x, fitness, state, params
    ) -> Tuple[chex.Array, chex.Array, OptState]:
        """Pmapped tell update call over multiple devices."""
        fitness = pmap_zscore(fitness)

        def calc_per_device_grad(x, fitness, mean, sigma):
            # Reconstruct noise from last mean/std estimates
            noise = (x - mean) / sigma
            theta_grad = 1.0 / (self.popsize * sigma) * jnp.dot(noise.T, fitness)
            return jax.lax.pmean(theta_grad, axis_name="p")

        theta_grad = jax.pmap(calc_per_device_grad, axis_name="p")(
            x, fitness, state.mean, state.sigma
        )

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = jax.pmap(self.optimizer.step)(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = jax.pmap(self.optimizer.update)(opt_state, params.opt_params)
        sigma = jax.pmap(exp_decay)(state.sigma, params.sigma_decay, params.sigma_limit)
        return mean, sigma, opt_state

    def single_tell(
        self, x, fitness, state, params
    ) -> Tuple[chex.Array, chex.Array, OptState]:
        """Single device tell update."""
        fitness = (fitness - jnp.mean(fitness)) / (jnp.std(fitness) + 1e-10)
        # Reconstruct noise from last mean/std estimates
        noise = (x - state.mean) / state.sigma
        theta_grad = 1.0 / (self.popsize * state.sigma) * jnp.dot(noise.T, fitness)

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)
        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)
        return mean, sigma, opt_state


def pmap_zscore(fitness: chex.Array) -> chex.Array:
    """Pmappable version of z-scoring of fitness scores."""

    def zscore(fit: chex.Array) -> chex.Array:
        all_mean = jax.lax.pmean(fit, axis_name="p").mean()
        diff = fit - all_mean
        std = jnp.sqrt(jax.lax.pmean(diff**2, axis_name="p").mean())
        return diff / (std + 1e-10)

    out = jax.pmap(zscore, axis_name="p")(fitness)
    return out


if __name__ == "__main__":
    """Run a simple example of a convex quadratic."""
    print(jax.devices())
    num_gens = 150
    popsize = int(32768 / 1)
    num_dims = 2000
    n_devices = 1

    def run_es(lrate_decay=0.99, sigma_decay=0.99, opt_name="adam"):
        rng = jax.random.PRNGKey(0)
        strategy = OpenES(
            popsize,
            num_dims,
            opt_name=opt_name,
            lrate_init=0.05,
            lrate_decay=lrate_decay,
            sigma_decay=sigma_decay,
            n_devices=n_devices,
        )
        es_params = strategy.default_params
        es_state = strategy.initialize(rng, es_params)
        print("Mean shape", es_state.mean.shape)
        print("Opt State", es_state.opt_state)
        print("ES Params", es_params)
        x, es_state = strategy.ask(rng, es_state, es_params)
        print("Solution shape", x.shape)

        def sphere(x):
            return jnp.sum(x**2)

        if n_devices > 1:
            psphere = jax.pmap(jax.vmap(sphere))
        else:
            psphere = jax.vmap(sphere)

        fitness = psphere(x)
        print("Fitness shape", fitness.shape)
        print(es_state.mean[0])
        es_state = strategy.tell(x, fitness, es_state, es_params)
        print(es_state.mean[0])

        all_fitness = []
        for i in range(num_gens):
            x, es_state = strategy.ask(rng, es_state, es_params)
            fitness = psphere(x)
            es_state = strategy.tell(x, fitness, es_state, es_params)
            print(i + 1, fitness.mean())
            all_fitness.append(fitness.mean())
        return all_fitness

    import matplotlib.pyplot as plt

    all_fitness100 = run_es(lrate_decay=1.0, sigma_decay=1.0, opt_name="adam")
    all_fitness099 = run_es(lrate_decay=0.99, sigma_decay=0.99, opt_name="adam")
    all_fitness095 = run_es(lrate_decay=0.95, sigma_decay=0.95, opt_name="adam")
    plt.plot(all_fitness100, label="lrate decay = 1.00, sigma_decay = 1.00")
    plt.plot(all_fitness099, label="lrate decay = 0.99, sigma_decay = 0.99")
    plt.plot(all_fitness095, label="lrate decay = 0.95, sigma_decay = 0.95")
    plt.xlabel("Generations")
    plt.ylabel("Mean Population Fitness")
    plt.title(f"{num_dims}D Quadratic - {popsize} Pop - Lrate 0.05, Sigma 0.04 - Adam")
    plt.legend()
    plt.ylim(0, 500)
    plt.savefig("quadratic_adam.png", dpi=300)
