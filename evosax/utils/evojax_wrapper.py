import jax
import jax.numpy as jnp
from evojax.algo.base import NEAlgorithm

from evosax import Strategy

from ..types import Fitness, Population, Solution


class Evosax2JAX_Wrapper(NEAlgorithm):
    """Wrapper for evosax-style ES for EvoJAX deployment."""

    def __init__(
        self,
        evosax_strategy: Strategy,
        params: dict = {},
        seed: int = 0,
    ):
        self.es = evosax_strategy
        self.params = self.es.default_params.replace(**params)

        self.key = jax.random.key(seed)
        self.key, key_init = jax.random.split(self.key)
        self.state = self.es.init(key_init, self.params)

    def ask(self) -> Population:
        """Ask strategy for next set of solution candidates to evaluate."""
        self.key, key_ask = jax.random.split(self.key)
        self.population, self.state = self.es.ask(key_ask, self.state, self.params)
        return self.population

    def tell(self, fitness: Fitness) -> None:
        """Tell strategy about most recent fitness evaluations."""
        self.state = self.es.tell(self.population, fitness, self.state, self.params)

    @property
    def best_solution(self) -> Solution:
        """Return set of mean/best parameters."""
        return self.state.mean

    @best_solution.setter
    def best_solution(self, best_solution: Solution) -> None:
        """Update the best parameters stored internally."""
        self.state = self.state.replace(mean=jnp.array(best_solution))

    @property
    def solution(self):
        """Get evaluation parameters for current ES state."""
        return self.es.get_eval_solution(self.state)
