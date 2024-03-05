from typing import Optional, Tuple, List
from flax import linen as nn
import jax.numpy as jnp
import chex
from .shared import expand_mask, scaled_dot_product, MLP, PositionalEncoding


class MultiheadPerceiver(nn.Module):
    num_latents: int
    latent_dim: int
    embed_dim: int
    num_heads: int
    dropout_prob: float = 0.0
    use_bias: bool = False
    out_att_maps: bool = False

    def setup(self):
        self.q_proj = nn.Dense(
            features=self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            use_bias=self.use_bias,
        )
        self.kv_proj = nn.Dense(
            features=2 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            use_bias=self.use_bias,
        )
        self.out_proj = nn.Dense(
            features=self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            use_bias=self.use_bias,
        )
        self.latents = self.param(
            "latents",
            nn.initializers.normal(),
            (
                self.num_latents,
                self.latent_dim,
            ),
        )
        self.attn_dropout = nn.Dropout(self.dropout_prob)
        self.resid_dropout = nn.Dropout(self.dropout_prob)

    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
        train: bool = True,
    ) -> Tuple[chex.Array, chex.Array]:
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        q = self.q_proj(self.latents)
        q = jnp.repeat(q.reshape(1, *q.shape), batch_size, axis=0)
        q = q.reshape(batch_size, self.num_latents, self.num_heads, -1)
        q = q.transpose(0, 2, 1, 3)

        kv = self.kv_proj(x)
        kv = kv.reshape(batch_size, seq_length, self.num_heads, -1)
        kv = kv.transpose(0, 2, 1, 3)
        k, v = jnp.array_split(kv, 2, axis=-1)

        attention = scaled_dot_product(q, k, mask)
        attention = self.attn_dropout(attention, deterministic=not train)
        values = jnp.matmul(attention, v)
        values = values.transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, self.num_latents, self.embed_dim)
        out = self.out_proj(values)
        out = self.resid_dropout(out, deterministic=not train)
        if self.out_att_maps:
            return out, attention
        else:
            return out, None


class PerceiverBlock(nn.Module):
    num_latents: int
    latent_dim: int
    num_heads: int
    embed_dim: int
    dropout_prob: float
    use_bias: bool
    out_att_maps: bool

    def setup(self):
        self.ln_1 = nn.LayerNorm(use_bias=self.use_bias)
        self.perceive = MultiheadPerceiver(
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_prob=self.dropout_prob,
            use_bias=self.use_bias,
            out_att_maps=self.out_att_maps,
        )
        self.ln_2 = nn.LayerNorm(use_bias=self.use_bias)
        self.mlp = MLP(self.embed_dim, self.dropout_prob, self.use_bias)

    def __call__(
        self, x: chex.Array, mask: Optional[chex.Array] = None, train: bool = True
    ) -> Tuple[chex.Array, chex.Array]:
        attn_out, attn = self.perceive(self.ln_1(x), mask, train)
        x = self.mlp(self.ln_2(attn_out), train)
        return x, attn


class PerceiverEncoder(nn.Module):
    num_latents: int
    latent_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout_prob: float = 0.0
    input_dropout_prob: float = 0.0
    use_bias: bool = False
    out_att_maps: bool = False

    def setup(self):
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.positional_encoding = PositionalEncoding(self.embed_dim)
        self.perceiver = [
            PerceiverBlock(
                num_latents=self.num_latents,
                latent_dim=self.latent_dim,
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                dropout_prob=self.dropout_prob,
                use_bias=self.use_bias,
                out_att_maps=self.out_att_maps,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
        add_positional_encoding: bool = True,
        train: bool = True,
    ) -> Tuple[chex.Array, List[chex.Array]]:
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.input_dropout(x, deterministic=not train)
        # Loop over transformer blocks and collect attention maps
        attn_maps = []
        for layer in self.perceiver:
            x, attn = layer(x, mask, train)
            attn_maps.append(attn)
        return x, attn_maps
