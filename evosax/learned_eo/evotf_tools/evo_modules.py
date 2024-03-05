from typing import List, Tuple
from functools import partial
import jax.numpy as jnp
from flax import linen as nn
import chex
from .attention import AttentionEncoder
from .perceiver import PerceiverEncoder


class CompressionPerceiver(nn.Module):
    num_latents: int
    latent_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int = 1
    dropout_prob: float = 0.0
    input_dropout_prob: float = 0.0
    use_bias: bool = False
    out_att_maps: bool = False

    def setup(self):
        self.cross_attn_population = partial(
            PerceiverEncoder,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_prob=self.dropout_prob,
            input_dropout_prob=self.input_dropout_prob,
            use_bias=self.use_bias,
            out_att_maps=self.out_att_maps,
        )
        self.lift_cross = nn.vmap(
            self.cross_attn_population,
            variable_axes={"params": None},
            split_rngs={"params": False, "dropout": True},
            in_axes=(0, None, None, None),
            out_axes=0,
        )

    @nn.compact
    def __call__(
        self, x: chex.Array, train: bool = False
    ) -> Tuple[chex.Array, List[chex.Array]]:
        x = x.transpose(1, 0, 2, 3)
        out, att = self.lift_cross(name="CompressionPerceiver")(
            x,
            None,
            False,
            train,
        )
        out = out.transpose(1, 0, 2, 3)
        if self.out_att_maps:
            att = [jnp.array(a).transpose(1, 0, 2, 3, 4) for a in att]
        return out, att


class SolutionPerceiver(nn.Module):
    num_latents: int
    latent_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int = 1
    dropout_prob: float = 0.0
    input_dropout_prob: float = 0.0
    use_bias: bool = False
    out_att_maps: bool = False

    def setup(self):
        self.cross_attn_population = partial(
            CompressionPerceiver,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_prob=self.dropout_prob,
            input_dropout_prob=self.input_dropout_prob,
            use_bias=self.use_bias,
            out_att_maps=self.out_att_maps,
        )
        self.lift_cross = nn.vmap(
            self.cross_attn_population,
            variable_axes={"params": None},
            split_rngs={"params": False, "dropout": False},
            in_axes=(0, None),
            out_axes=0,
        )

    @nn.compact
    def __call__(
        self, x: chex.Array, train: bool = False
    ) -> Tuple[chex.Array, List[chex.Array]]:
        x = x.transpose(3, 0, 1, 2, 4)
        out, att = self.lift_cross(name="SolutionPerceiver")(
            x,
            train,
        )
        out = out.transpose(1, 2, 3, 0, 4)
        if self.out_att_maps:
            att = [jnp.array(a).transpose(1, 2, 0, 3, 4, 5) for a in att]
        return out, att


class DistributionAttention(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int = 1
    dropout_prob: float = 0.0
    input_dropout_prob: float = 0.0
    use_bias: bool = False
    out_att_maps: bool = False

    def setup(self):
        self.transformer = partial(
            AttentionEncoder,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            dropout_prob=self.dropout_prob,
            input_dropout_prob=self.input_dropout_prob,
            use_bias=self.use_bias,
            out_att_maps=self.out_att_maps,
        )

        self.lift_att = nn.vmap(
            self.transformer,
            variable_axes={"params": None},
            split_rngs={"params": False, "dropout": False},
            in_axes=(0, None, None, None),
            out_axes=0,
        )

    @nn.compact
    def __call__(
        self, x: chex.Array, train: bool = True
    ) -> Tuple[chex.Array, List[chex.Array]]:
        x = x.transpose(1, 0, 2, 3)
        out, att = self.lift_att(name="DistributionAttention")(x, None, False, train)
        out = out.transpose(1, 0, 2, 3)
        if self.out_att_maps:
            att = jnp.array(att).transpose(2, 1, 0, 3, 4, 5)
        return out, att


class DistributionUpdateNetwork(nn.Module):
    embed_dim: int
    dropout_prob: float = 0.0
    use_bias: bool = False

    def setup(self):
        self.output_net = [
            nn.Dense(
                features=self.embed_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                use_bias=self.use_bias,
            ),
            nn.LayerNorm(self.use_bias),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(
                features=2,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                use_bias=self.use_bias,
            ),
        ]

    @nn.compact
    def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
        out = x
        for l in self.output_net:
            out = (
                l(out)
                if not isinstance(l, nn.Dropout)
                else l(out, deterministic=not train)
            )
        return out
