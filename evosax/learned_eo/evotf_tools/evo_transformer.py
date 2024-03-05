from functools import partial
import jax.numpy as jnp
from flax import linen as nn
import chex
from .evo_modules import (
    CompressionPerceiver,
    SolutionPerceiver,
    DistributionAttention,
    DistributionUpdateNetwork,
)
from .attention import AttentionEncoder


class EvoTransformer(nn.Module):
    embed_dim: int
    num_heads: int
    num_latents: int
    latent_dim: int
    num_layers: int = 1
    dropout_prob: float = 0.0
    input_dropout_prob: float = 0.0
    use_bias: bool = False
    use_fitness_encoder: bool = True
    use_dist_encoder: bool = True
    use_crossd_encoder: bool = True
    out_att_maps: bool = False

    def setup(self):
        self.sol_perceiver = SolutionPerceiver(
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
        self.fit_perceiver = CompressionPerceiver(
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
        self.dist_encoder = DistributionAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_prob=self.dropout_prob,
            input_dropout_prob=self.input_dropout_prob,
            use_bias=self.use_bias,
            out_att_maps=self.out_att_maps,
        )
        self.crossd_encoder = CompressionPerceiver(
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=1,  # For now only with a single layer possible
            dropout_prob=self.dropout_prob,
            input_dropout_prob=self.input_dropout_prob,
            use_bias=self.use_bias,
            out_att_maps=self.out_att_maps,
        )
        self.transformer_backbone = partial(
            AttentionEncoder,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_prob=self.dropout_prob,
            input_dropout_prob=self.input_dropout_prob,
            use_bias=self.use_bias,
            out_att_maps=self.out_att_maps,
        )
        self.dist_update = partial(
            DistributionUpdateNetwork,
            embed_dim=self.embed_dim,
            dropout_prob=self.dropout_prob,
            use_bias=self.use_bias,
        )
        self.lift_transformer = nn.vmap(
            self.transformer_backbone,
            variable_axes={"params": None},
            split_rngs={"params": False, "dropout": False},
            in_axes=(0, None, None, None),
            out_axes=0,
        )
        self.lift_distribution = nn.vmap(
            self.dist_update,
            variable_axes={"params": None},
            split_rngs={"params": False, "dropout": False},
            in_axes=(0, None),
            out_axes=0,
        )

    @nn.compact
    def __call__(
        self,
        solution_features: chex.Array,
        fitness_features: chex.Array,
        dist_features: chex.Array,
        mask=None,
        add_positional_encoding=True,
        train=False,
        verbose=False,
    ):
        batch_size, seq_len, popsize, num_dims, feature_dim = solution_features.shape
        sol_encoding, sol_att = self.sol_perceiver(solution_features, train)
        if self.use_fitness_encoder:
            fit_encoding, fit_att = self.fit_perceiver(fitness_features, train)
        if self.use_dist_encoder:
            dist_encoding, dist_att = self.dist_encoder(dist_features, train)
        if self.use_crossd_encoder:
            crossd_encoding, crossd_att = self.crossd_encoder(dist_features, train)

        if verbose:
            print("Solution encoding shape:", sol_encoding.shape)
            if self.use_fitness_encoder:
                print("Fitness encoding shape:", fit_encoding.shape)
            if self.use_dist_encoder:
                print("Dist encoding shape:", dist_encoding.shape)
            if self.use_crossd_encoder:
                print("Crossd encoding shape:", crossd_encoding.shape)

        sol_encoding = sol_encoding.transpose(3, 0, 1, 2, 4)
        if verbose:
            print("Solution encoding shape after reshape:", sol_encoding.shape)

        if self.use_fitness_encoder:
            fit_encoding = jnp.repeat(
                fit_encoding.reshape(1, *fit_encoding.shape), num_dims, axis=0
            )
            if verbose:
                print("Fitness encoding shape after reshape:", fit_encoding.shape)

        if self.use_dist_encoder:
            dist_encoding = dist_encoding.transpose(2, 0, 1, 3)
            if verbose:
                print("Dist encoding shape after reshape:", dist_encoding.shape)

        if self.use_crossd_encoder:
            crossd_encoding = jnp.repeat(
                crossd_encoding.reshape(1, *crossd_encoding.shape), num_dims, axis=0
            )
            if verbose:
                print("Crossd encoding shape after reshape:", crossd_encoding.shape)

        combined_encoding = sol_encoding
        if self.use_fitness_encoder:
            combined_encoding = jnp.concatenate(
                [combined_encoding, fit_encoding], axis=-1
            )
        combined_encoding = combined_encoding.reshape(num_dims, batch_size, seq_len, -1)
        if verbose:
            print("Combined encoding shape:", combined_encoding.shape)

        if self.use_dist_encoder:
            combined_encoding = jnp.concatenate(
                [combined_encoding, dist_encoding], axis=-1
            )
            if verbose:
                print("Combined encoding shape after dist:", combined_encoding.shape)

        if self.use_crossd_encoder:
            crossd_encoding = crossd_encoding.reshape(num_dims, batch_size, seq_len, -1)
            combined_encoding = jnp.concatenate(
                [combined_encoding, crossd_encoding], axis=-1
            )
            if verbose:
                print("Combined encoding shape after crossd:", combined_encoding.shape)

        out, mhsa_att = self.lift_transformer(name="DimBatchedTransformer")(
            combined_encoding, mask, add_positional_encoding, train
        )
        if self.out_att_maps:
            mhsa_att = jnp.array(mhsa_att).transpose(2, 1, 0, 3, 4, 5)
        if verbose:
            print("Transformer output shape:", out.shape)

        distrib_out = self.lift_distribution(name="DistributionUpdate")(out, train)
        distrib_out = distrib_out.transpose(3, 1, 2, 0)
        if verbose:
            print("Distribution output shape:", distrib_out.shape)

        # Collect all attention maps
        all_att_out = {"solution": sol_att, "time": mhsa_att}
        if self.use_fitness_encoder:
            all_att_out["fitness"] = fit_att
        if self.use_dist_encoder:
            all_att_out["distribution"] = dist_att
        if self.use_crossd_encoder:
            all_att_out["cross_dim"] = crossd_att
        return distrib_out, all_att_out
