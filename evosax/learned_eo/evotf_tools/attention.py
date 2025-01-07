from typing import Tuple, Optional, List
from flax import linen as nn
import jax.numpy as jnp
import chex
from .shared import scaled_dot_product, expand_mask, MLP, PositionalEncoding


class MultiheadAttention(nn.Module):
    embed_dim: int
    num_heads: int
    dropout_prob: float = 0.0
    use_bias: bool = False
    out_att_maps: bool = False

    def setup(self):
        self.qkv_proj = nn.Dense(
            features=3 * self.embed_dim,
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
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        attention = scaled_dot_product(q, k, mask)
        attention = self.attn_dropout(attention, deterministic=not train)
        values = jnp.matmul(attention, v)
        values = values.transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, embed_dim)
        out = self.out_proj(values)
        out = self.resid_dropout(out, deterministic=not train)
        if self.out_att_maps:
            return out, attention
        else:
            return out, None


class AttentionBlock(nn.Module):
    num_heads: int
    embed_dim: int
    dropout_prob: float
    use_bias: bool
    out_att_maps: bool

    def setup(self):
        self.ln_1 = nn.LayerNorm(use_bias=self.use_bias)
        self.attn = MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            self.dropout_prob,
            self.use_bias,
            self.out_att_maps,
        )
        self.ln_2 = nn.LayerNorm(use_bias=self.use_bias)
        self.mlp = MLP(self.embed_dim, self.dropout_prob, self.use_bias)

    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
        train: bool = True,
    ) -> Tuple[chex.Array, chex.Array]:
        attn_out, attn = self.attn(self.ln_1(x), mask, train)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x), train)
        return x, attn


class AttentionEncoder(nn.Module):
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
        self.transformer = [
            AttentionBlock(
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
        train=True,
    ) -> Tuple[chex.Array, List[chex.Array]]:
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.input_dropout(x, deterministic=not train)
        # Loop over transformer blocks and collect attention maps
        attn_maps = []
        for layer in self.transformer:
            x, attn = layer(x, mask, train)
            attn_maps.append(attn)
        return x, attn_maps
