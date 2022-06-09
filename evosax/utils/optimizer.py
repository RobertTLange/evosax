import jax
import jax.numpy as jnp
import chex
from typing import Optional, Dict, Tuple
from flax import struct

# TODO: Add gradient clipping - select leads to more compute
# "use_clip_by_global_norm": False,
# "clip_global_norm": 5,
# "use_clip_by_value": False,
# "clip_value": 5,


@struct.dataclass
class OptState:
    lrate: float
    m: chex.Array
    v: Optional[chex.Array] = None
    gen_counter: int = 0


@struct.dataclass
class OptParams:
    lrate_init: float = 0.01
    lrate_decay: float = 0.999
    lrate_limit: float = 0.001
    momentum: Optional[float] = None
    beta_1: Optional[float] = None
    beta_2: Optional[float] = None
    eps: Optional[float] = None
    max_speed: Optional[float] = None


class Optimizer(object):
    def __init__(self, num_dims: int):
        """Simple JAX-Compatible Optimizer Class."""
        self.num_dims = num_dims

    @property
    def default_params(self) -> OptParams:
        """Return shared and optimizer-specific default parameters."""
        return OptParams(**self.params_opt)

    def initialize(self, params: OptParams) -> OptState:
        """Initialize the optimizer state."""
        return self.initialize_opt(params)

    def step(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: OptState,
        params: OptParams,
    ) -> [chex.Array, OptState]:
        """Perform a gradient-based update step."""
        return self.step_opt(mean, grads, state, params)

    def update(self, state: OptState, params: OptParams) -> OptState:
        """Exponentially decay the learning rate if desired."""
        lrate = state.lrate * params.lrate_decay
        lrate = jnp.maximum(lrate, params.lrate_limit)
        return state.replace(lrate=lrate)

    @property
    def params_opt(self) -> OptParams:
        """Optimizer-specific hyperparameters."""
        raise NotImplementedError

    def initialize_opt(self, params: OptParams) -> OptState:
        """Optimizer-specific initialization of optimizer state."""
        raise NotImplementedError

    def step_opt(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: OptState,
        params: OptParams,
    ) -> Tuple[chex.Array, OptState]:
        """Optimizer-specific step to update parameter estimates."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, num_dims: int):
        """Simple JAX-Compatible SGD + Momentum optimizer."""
        super().__init__(num_dims)
        self.opt_name = "sgd"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default SGD+Momentum parameters."""
        return {
            "momentum": 0.9,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the momentum trace of the optimizer."""
        return OptState(m=jnp.zeros(self.num_dims), lrate=params.lrate_init)

    def step_opt(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> Tuple[chex.Array, OptState]:
        """Perform a simple SGD + Momentum step."""
        m = grads + params.momentum * state.m
        mean_new = mean - state.lrate * state.m
        return mean_new, state.replace(m=m, gen_counter=state.gen_counter + 1)


class Adam(Optimizer):
    def __init__(self, num_dims: int):
        """JAX-Compatible Adam Optimizer (Kingma & Ba, 2015)
        Reference: https://arxiv.org/abs/1412.6980"""
        super().__init__(num_dims)
        self.opt_name = "adam"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default Adam parameters."""
        return {
            "beta_1": 0.99,
            "beta_2": 0.999,
            "eps": 1e-8,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the m, v trace of the optimizer."""
        return OptState(
            m=jnp.zeros(self.num_dims),
            v=jnp.zeros(self.num_dims),
            lrate=params.lrate_init,
        )

    def step_opt(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: OptState,
        params: OptParams,
    ) -> Tuple[chex.Array, OptState]:
        """Perform a simple Adam GD step."""
        m = (1 - params.beta_1) * grads + params.beta_1 * state.m
        v = (1 - params.beta_2) * (grads ** 2) + params.beta_2 * state.v
        mhat = m / (1 - params.beta_1 ** (state.gen_counter + 1))
        vhat = v / (1 - params.beta_2 ** (state.gen_counter + 1))
        mean_new = mean - state.lrate * mhat / (jnp.sqrt(vhat) + params.eps)
        return mean_new, state.replace(
            m=m, v=v, gen_counter=state.gen_counter + 1
        )


class RMSProp(Optimizer):
    def __init__(self, num_dims: int):
        """JAX-Compatible RMSProp Optimizer (Hinton et al., 2012)
        Reference: https://tinyurl.com/2sbbcnrv"""
        super().__init__(num_dims)
        self.opt_name = "rmsprop"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default RMSProp parameters."""
        return {
            "momentum": 0.9,
            "beta_1": 0.99,
            "eps": 1e-8,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the m, v trace of the optimizer."""
        return OptState(
            m=jnp.zeros(self.num_dims),
            v=jnp.zeros(self.num_dims),
            lrate=params.lrate_init,
        )

    def step_opt(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: OptState,
        params: OptParams,
    ) -> Tuple[chex.Array, OptState]:
        """Perform a simple RMSprop GD step."""
        v = (1 - params.beta_1) * (grads ** 2) + params.beta_1 * state.v
        m = params.momentum * state.m + grads / (jnp.sqrt(v) + params.eps)
        mean_new = mean - state.lrate * m
        return mean_new, state.replace(
            m=m, v=v, gen_counter=state.gen_counter + 1
        )


class ClipUp(Optimizer):
    def __init__(self, num_dims: int):
        """JAX-Compatible ClipUp Optimizer (Toklu et al., 2020)
        Reference: https://arxiv.org/abs/2008.02387"""
        super().__init__(num_dims)
        self.opt_name = "clipup"

    @property
    def params_opt(self) -> Dict[str, float]:
        """Return default ClipUp parameters."""
        return {
            "lrate_init": 0.15,
            "lrate_decay": 0.999,
            "lrate_limit": 0.05,
            "max_speed": 0.3,
            "momentum": 0.9,
        }

    def initialize_opt(self, params: OptParams) -> OptState:
        """Initialize the momentum trace of the optimizer."""
        return OptState(m=jnp.zeros(self.num_dims), lrate=params.lrate_init)

    def step_opt(
        self,
        mean: chex.Array,
        grads: chex.Array,
        state: OptState,
        params: OptParams,
    ) -> Tuple[chex.Array, OptState]:
        """Perform a ClipUp step. mom = 0.9, lrate = vmax/2, vmax = small"""
        # Normalize length of gradients - vmax & alpha control max step magnitude
        grad_magnitude = jnp.sqrt(jnp.sum(grads * grads))
        gradient = grads / (grad_magnitude + 1e-08)
        step = gradient * state.lrate
        velocity = params.momentum * state.m + step

        def clip(velocity: chex.Array, max_speed: float):
            """Rescale clipped velocities."""
            vel_magnitude = jnp.sqrt(jnp.sum(velocity * velocity))
            ratio_scale = vel_magnitude > max_speed
            scaled_vel = velocity * (max_speed / (vel_magnitude + 1e-08))
            x_out = jax.lax.select(ratio_scale, scaled_vel, velocity)
            return x_out

        # Clip the update velocity and apply the update
        m = clip(velocity, params.max_speed)
        mean_new = mean - state.lrate * m
        return mean_new, state.replace(m=m)
