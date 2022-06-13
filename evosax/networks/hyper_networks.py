from flax import linen as nn
from flax.traverse_util import unflatten_dict
from typing import Callable


class HyperNetworkMLP(nn.Module):
    """2-Layer MLP Hypernetwork as in Ha et al. 2016."""

    raw_network_shapes: dict
    num_latent_units: int  # Dim of latents z
    num_hidden_units: int  # Dim of hidden layer a
    kernel_init: Callable = nn.initializers.lecun_normal()

    def setup(self):
        # Define the latents Z for each layer
        self.latents = [
            self.param(
                f"latent_{i}", self.kernel_init, (self.num_latent_units, 1)
            )
            for i, s in enumerate(self.raw_network_shapes.keys())
        ]
        # Get max number of out units - use to embed all latents
        # Afterwards - subselect required number of columns
        max_out_units = max(sum(list(self.raw_network_shapes.values()), ()))
        self.max_out_units = int(max_out_units)

        # Add shape hack for bias terms - unsqueeze for matmul and later squeeze away
        network_shapes = {}
        for k, v in self.raw_network_shapes.items():
            if len(v) == 1:
                network_shapes[k] = v + (1,)
            else:
                network_shapes[k] = v
        self.network_shapes = network_shapes

    @nn.compact
    def __call__(self):
        a_latent = []
        # Perform projection for all in dims - loop over modules
        for i, s in enumerate(self.network_shapes.keys()):
            a = nn.DenseGeneral(
                (self.num_hidden_units, self.network_shapes[s][0])
            )
            stacked_d = a(self.latents[i].squeeze())
            # Shape stacked_d - (num_hidden_units, out_dim of layer)
            a_latent.append(stacked_d)

        # Perform weight matrix readout for each module using shared dense
        W_out = nn.DenseGeneral(self.max_out_units, axis=0)
        weight_matrices = {}
        for i, s in enumerate(self.network_shapes.keys()):
            w_t = W_out(a_latent[i])
            # Subselect out columns based on required shape
            weight_matrices[s] = w_t[:, : self.network_shapes[s][-1]].reshape(
                self.raw_network_shapes[s]
            )
        return unflatten_dict(
            {tuple(k.split("/")): v for k, v in weight_matrices.items()}
        )
