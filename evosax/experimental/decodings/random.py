import chex
import jax

from ...utils.helpers import get_ravel_fn
from .decoder import Decoder


class RandomDecoder(Decoder):
    def __init__(
        self,
        num_encoding_dims: int,
        placeholder_params: chex.ArrayTree | chex.Array,
        key: jax.Array = jax.random.key(0),
        rademacher: bool = False,
    ):
        """Random Projection Decoder (Gaussian/Rademacher random matrix)."""
        super().__init__(num_encoding_dims, placeholder_params)
        self.rademacher = rademacher

        self.ravel_params, self.unravel_params = get_ravel_fn(placeholder_params)
        flat_params = self.ravel_params(placeholder_params)
        total_params = flat_params.size

        # Sample a random matrix - Gaussian or Rademacher (+1/-1)
        if not self.rademacher:
            self.project_matrix = jax.random.normal(
                key, (self.num_encoding_dims, total_params)
            )
        else:
            self.project_matrix = jax.random.rademacher(
                key, (self.num_encoding_dims, total_params)
            )
        print(f"RandomDecoder: Encoding parameters to optimize - {num_encoding_dims}")

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for random projection case."""
        # 1. Project parameters to raw dimensionality using pre-sampled matrix
        project_x = (
            x @ self.project_matrix
        )  # (popsize, num_enc_dim) x (num_enc_dim, num_dims)
        # 2. Reshape
        x_reshaped = jax.vmap(self.unravel_params)(project_x)
        return x_reshaped

    def reshape_single(self, x: chex.Array) -> chex.ArrayTree:
        """Reshape a single flat vector using random projection matrix."""
        x_re = x.reshape(1, self.num_encoding_dims)
        # 1. Project parameters to raw dimensionality using pre-sampled matrix
        project_x = (x_re @ self.project_matrix).squeeze()
        # 2. Reshape
        x_reshaped = self.unravel_params(project_x)
        return x_reshaped
