"""Custom optimizers for evolutionary algorithms.

This module provides custom optimization algorithms that are not available in optax.
These optimizers are designed specifically for evolutionary algorithms and can be
used as drop-in replacements for optax optimizers in evosax algorithms.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class ScaleByClipUpState(NamedTuple):
    velocity: optax.Updates
    count: jnp.ndarray


def scale_by_clipup(
    max_velocity_ratio: float = 2.0,
    momentum: float = 0.9,
) -> optax.GradientTransformation:
    """Core ClipUp gradient scaling transformation.

    This transformation implements the core ClipUp steps:
    1. Update velocity with momentum.
    2. Clip velocity if its norm exceeds max_velocity_ratio.
    3. Return the clipped velocity as updates.

    Note 1: This does not apply gradient normalization, which should be handled by a
    preceding transformation (e.g., optax.normalize_by_update_norm()).

    Note 2: This does not apply the learning rate, which should be handled by a
    following transformation (e.g., optax.scale_by_learning_rate(learning_rate)).

    Args:
        max_velocity_ratio: Maximum velocity magnitude before learning rate scaling.
        momentum: Momentum coefficient (decay rate for velocity).

    Returns:
        An optax.GradientTransformation that computes the clipped velocity direction.

    """

    def init_fn(params):
        return ScaleByClipUpState(
            velocity=jax.tree.map(jnp.zeros_like, params),
            count=jnp.zeros([], dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        del params  # Unused

        # Update velocity with momentum
        new_velocity = jax.tree.map(
            lambda v, u: momentum * v + u, state.velocity, updates
        )

        # Clip velocity if its norm exceeds max_velocity_ratio
        velocity_norm = optax.global_norm(new_velocity)
        scale = jnp.where(
            velocity_norm > max_velocity_ratio, max_velocity_ratio / velocity_norm, 1.0
        )
        clipped_velocity = jax.tree.map(lambda v: v * scale, new_velocity)

        # Update state with new velocity and increment count
        count_inc = state.count + 1
        return clipped_velocity, ScaleByClipUpState(
            velocity=clipped_velocity, count=count_inc
        )

    return optax.GradientTransformation(init_fn, update_fn)


def clipup(
    learning_rate: float,
    max_velocity: float = 0.1,
    momentum: float = 0.9,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """ClipUp optimizer.

    This transformation implements the ClipUp steps:
    1. Normalizes the gradient globally.
    2. Applies momentum to update velocity.
    3. Clip velocity if its norm exceeds max_velocity_ratio.
    4. Return the clipped velocity as updates.
    5. Scales by learning_rate.

    Args:
        learning_rate: Learning rate to scale the updates.
        max_velocity: Maximum step size magnitude (||updates|| <= max_velocity).
        momentum: Momentum coefficient.
        eps: Small constant to prevent division by zero.

    Returns:
        An optax.GradientTransformation representing the ClipUp optimizer.

    Reference:
        Toklu et al., "ClipUp: A Simple and Powerful Optimizer for Distribution-based
        Policy Evolution", 2020.
        https://arxiv.org/abs/2008.02387

    """
    return optax.chain(
        optax.normalize_by_update_norm(eps=eps),
        scale_by_clipup(
            max_velocity_ratio=max_velocity / learning_rate,
            momentum=momentum,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )
