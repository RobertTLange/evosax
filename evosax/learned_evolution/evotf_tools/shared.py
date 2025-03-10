import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def scaled_dot_product(
    q: jax.Array,
    k: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    return attention


def expand_mask(mask: jax.Array) -> jax.Array:
    assert mask.ndim >= 2
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, axis=1)
    while mask.ndim < 4:
        mask = jnp.expand_dims(mask, axis=0)
    return mask


class MLP(nn.Module):
    embed_dim: int
    dropout_prob: float
    use_bias: bool

    def setup(self):
        self.linear = [
            nn.Dense(
                features=4 * self.embed_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                use_bias=self.use_bias,
            ),
            nn.gelu,
            nn.Dense(
                features=self.embed_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                use_bias=self.use_bias,
            ),
            nn.Dropout(self.dropout_prob),
        ]

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        for linear in self.linear:
            x = (
                linear(x)
                if not isinstance(linear, nn.Dropout)
                else linear(x, deterministic=not train)
            )
        return x


class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32)
            * -(np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.pe[:, : x.shape[1]]
        return x
