import chex


class Decoder:
    def __init__(
        self,
        num_encoding_dims: int,
        solution: chex.ArrayTree | chex.Array,
    ):
        self.num_encoding_dims = num_encoding_dims
        self.total_params = num_encoding_dims
        self.solution = solution

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        raise NotImplementedError

    def reshape_single(self, x: chex.Array) -> chex.ArrayTree:
        raise NotImplementedError
