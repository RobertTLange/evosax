import jax

from ...types import Population, Solution


class Decoder:
    def __init__(
        self,
        num_encoding_dims: int,
        solution: Solution,
    ):
        self.num_encoding_dims = num_encoding_dims
        self.total_params = num_encoding_dims
        self.solution = solution

    def reshape(self, solutions: Population) -> jax.Array:
        raise NotImplementedError

    def reshape_single(self, solution: Solution) -> jax.Array:
        raise NotImplementedError
