import jax
import chex
from typing import Union, Optional
from .decoder import Decoder
from ...core import ParameterReshaper


class RandomDecoder(Decoder):
    def __init__(
        self,
        num_encoding_dims: int,
        placeholder_params: Union[chex.ArrayTree, chex.Array],
        rng: chex.PRNGKey = jax.random.PRNGKey(0),
        rademacher: bool = False,
        n_devices: Optional[int] = None,
    ):
        """Random Projection Decoder (Gaussian/Rademacher random matrix)."""
        super().__init__(num_encoding_dims, placeholder_params, n_devices)
        self.rademacher = rademacher
        # Instantiate base reshaper class
        self.base_reshaper = ParameterReshaper(
            placeholder_params, n_devices, verbose=False
        )
        self.vmap_dict = self.base_reshaper.vmap_dict

        # Sample a random matrix - Gaussian or Rademacher (+1/-1)
        if not self.rademacher:
            self.project_matrix = jax.random.normal(
                rng, (self.num_encoding_dims, self.base_reshaper.total_params)
            )
        else:
            self.project_matrix = jax.random.rademacher(
                rng, (self.num_encoding_dims, self.base_reshaper.total_params)
            )
        print(
            "RandomDecoder: Encoding parameters to optimize -"
            f" {num_encoding_dims}"
        )

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for random projection case."""
        # 1. Project parameters to raw dimensionality using pre-sampled matrix
        project_x = (
            x @ self.project_matrix
        )  # (popsize, num_enc_dim) x (num_enc_dim, num_dims)
        # 2. Reshape using base reshaper class
        x_reshaped = self.base_reshaper.reshape(project_x)
        return x_reshaped

    def reshape_single(self, x: chex.Array) -> chex.ArrayTree:
        """Reshape a single flat vector using random projection matrix."""
        x_re = x.reshape(1, self.num_encoding_dims)
        # 1. Project parameters to raw dimensionality using pre-sampled matrix
        project_x = (x_re @ self.project_matrix).squeeze()
        # 2. Reshape using base reshaper class
        x_reshaped = self.base_reshaper.reshape_single(project_x)
        return x_reshaped
