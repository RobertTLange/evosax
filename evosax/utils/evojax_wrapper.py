import chex
import jax
import jax.numpy as jnp
from evojax.algo.base import NEAlgorithm
from evosax import Strategy


class Evosax2JAX_Wrapper(NEAlgorithm):
    """Wrapper for evosax-style ES for EvoJAX deployment."""

    def __init__(
        self,
        evosax_strategy: Strategy,
        param_size: int,
        pop_size: int,
        es_config: dict = {},
        es_params: dict = {},
        opt_params: dict = {},
        seed: int = 42,
    ):
        self.es = evosax_strategy(
            popsize=pop_size, num_dims=param_size, **es_config, **opt_params
        )
        self.es_params = self.es.default_params.replace(**es_params)
        self.pop_size = pop_size
        self.param_size = param_size
        self.rand_key = jax.random.PRNGKey(seed=seed)
        self.rand_key, init_key = jax.random.split(self.rand_key)
        self.es_state = self.es.initialize(init_key, self.es_params)

    def ask(self) -> chex.Array:
        """Ask strategy for next set of solution candidates to evaluate."""
        self.rand_key, ask_key = jax.random.split(self.rand_key)
        self.params, self.es_state = self.es.ask(
            ask_key, self.es_state, self.es_params
        )
        return self.params

    def tell(self, fitness: chex.Array) -> None:
        """Tell strategy about most recent fitness evaluations."""
        self.es_state = self.es.tell(
            self.params, fitness, self.es_state, self.es_params
        )

    @property
    def best_params(self) -> chex.Array:
        """Return set of mean/best parameters."""
        return jnp.array(self.es_state.mean, copy=True)

    @best_params.setter
    def best_params(self, params: chex.Array) -> None:
        """Update the best parameters stored internally."""
        self.es_state = self.es_state.replace(mean=jnp.array(params, copy=True))

    @property
    def solution(self):
        """Get evaluation parameters for current ES state."""
        return self.es.get_eval_params(self.es_state)
