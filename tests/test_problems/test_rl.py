"""Tests for reinforcement learning problems."""

from importlib.util import find_spec

import jax
import jax.numpy as jnp
import pytest
from evosax.problems import BraxProblem, GymnaxProblem
from evosax.problems.networks import MLP
from evosax.problems.rl.gymnax import State as GymnaxState

requires_gymnax = pytest.mark.skipif(
    find_spec("gymnax") is None, reason="gymnax is not installed"
)
requires_brax = pytest.mark.skipif(
    find_spec("brax") is None, reason="brax is not installed"
)


class ConstantPolicy:
    def apply(self, params, obs, key):
        return 0


class TerminalRewardEnvironment:
    def __init__(self, termination_kind):
        self.termination_kind = termination_kind

    def reset(self, key, params):
        return jnp.array(0.0), jnp.array(0)

    def step(self, key, state, action, params):
        next_state = state + 1
        terminated = jnp.array(self.termination_kind == "terminated")
        truncated = jnp.array(self.termination_kind == "truncated")
        return jnp.array(0.0), next_state, jnp.array(1.0), terminated, truncated, {}


@requires_gymnax
def test_gymnax_problem_init():
    """Test GymnaxProblem initialization with default settings."""
    policy = MLP(layer_sizes=(64, 64, 2))
    problem = GymnaxProblem(
        env_name="CartPole-v1", policy=policy, episode_length=100, num_rollouts=5
    )

    assert problem.env_name == "CartPole-v1"
    assert problem.episode_length == 100
    assert problem.num_rollouts == 5


@requires_gymnax
def test_gymnax_problem_sample():
    """Test GymnaxProblem solution sampling."""
    key = jax.random.key(0)
    policy = MLP(layer_sizes=(64, 64, 2))
    problem = GymnaxProblem(
        env_name="CartPole-v1", policy=policy, episode_length=100, num_rollouts=5
    )

    # Sample a solution
    solution = problem.sample(key)

    # Check that solution is a valid PyTree
    flat_params, _ = jax.flatten_util.ravel_pytree(solution)
    assert flat_params.ndim == 1


@requires_gymnax
def test_gymnax_problem_eval():
    """Test GymnaxProblem evaluation."""
    key = jax.random.key(0)
    policy = MLP(layer_sizes=(64, 64, 2))
    problem = GymnaxProblem(
        env_name="CartPole-v1", policy=policy, episode_length=100, num_rollouts=3
    )

    # Initialize state
    state = problem.init(key)

    # Create a batch of solutions using vmap
    population_size = 4
    keys = jax.random.split(key, population_size)

    # Create a batch of solutions
    solutions = jax.vmap(problem.sample)(keys)

    # Evaluate the solutions
    key_eval = jax.random.key(42)
    fitness, new_state, info = problem.eval(key_eval, solutions, state)

    # Check shape (population_size,)
    assert fitness.shape == (population_size,)
    assert new_state.counter == state.counter + 1


@pytest.mark.parametrize(
    "termination_kind",
    [
        pytest.param("terminated", id="terminated"),
        pytest.param("truncated", id="truncated"),
    ],
)
def test_gymnax_problem_masks_rewards_after_episode_end(termination_kind):
    """Test terminated and truncated episodes do not add autoreset rewards."""
    problem = GymnaxProblem.__new__(GymnaxProblem)
    problem.env = TerminalRewardEnvironment(termination_kind)
    problem.env_params = None
    problem.policy = ConstantPolicy()
    problem.episode_length = 4
    problem.use_normalize_obs = False

    fitness, _ = problem._rollout(jax.random.key(0), policy_params=None, state=None)

    assert fitness.item() == pytest.approx(1.0)


@requires_gymnax
def test_gymnax_problem_applies_environment_parameter_overrides():
    """Test GymnaxProblem applies parameter overrides to its environment."""
    problem = GymnaxProblem(
        env_name="CartPole-v1",
        policy=MLP(layer_sizes=(64, 64, 2)),
        env_params={"max_steps_in_episode": 3},
    )

    assert problem.env_params.max_steps_in_episode == 3
    assert problem.episode_length == 3


def test_gymnax_problem_updates_pytree_observation_statistics():
    """Test observation normalization statistics support PyTree observations."""
    observations = {
        "position": jnp.array([[[1.0], [3.0]]]),
        "velocity": jnp.array([[[2.0], [4.0]]]),
    }
    state = GymnaxState(
        counter=0,
        obs_mean=jax.tree.map(lambda leaf: jnp.zeros_like(leaf[0, 0]), observations),
        obs_std=jax.tree.map(lambda leaf: jnp.ones_like(leaf[0, 0]), observations),
        obs_var_sum=jax.tree.map(lambda leaf: jnp.zeros_like(leaf[0, 0]), observations),
        obs_counter=0,
        std_min=1e-6,
        std_max=1e6,
    )

    updated_state = jax.jit(
        lambda obs, current_state: GymnaxProblem.update_stats(None, obs, current_state)
    )(observations, state)

    assert updated_state.obs_counter == 2
    assert updated_state.obs_mean["position"].item() == pytest.approx(2.0)
    assert updated_state.obs_mean["velocity"].item() == pytest.approx(3.0)
    assert updated_state.obs_std["position"].item() == pytest.approx(1.0)
    assert updated_state.obs_std["velocity"].item() == pytest.approx(1.0)


@requires_brax
def test_brax_problem_init():
    """Test BraxProblem initialization with default settings."""
    policy = MLP(layer_sizes=(64, 64, 1))
    problem = BraxProblem(
        env_name="ant", policy=policy, episode_length=100, num_rollouts=5
    )

    assert problem.env_name == "ant"
    assert problem.episode_length == 100
    assert problem.num_rollouts == 5


@requires_brax
def test_brax_problem_sample():
    """Test BraxProblem solution sampling."""
    key = jax.random.key(0)
    policy = MLP(layer_sizes=(64, 64, 8))  # Ant has 8 actions
    problem = BraxProblem(
        env_name="ant", policy=policy, episode_length=100, num_rollouts=5
    )

    # Sample a solution
    solution = problem.sample(key)

    # Check that solution is a valid PyTree
    flat_params, _ = jax.flatten_util.ravel_pytree(solution)
    assert flat_params.ndim == 1


@requires_brax
def test_brax_problem_eval():
    """Test BraxProblem evaluation."""
    key = jax.random.key(0)
    policy = MLP(layer_sizes=(64, 64, 8))  # Ant has 8 actions
    problem = BraxProblem(
        env_name="ant", policy=policy, episode_length=100, num_rollouts=3
    )

    # Initialize state
    state = problem.init(key)

    # Create a batch of solutions using vmap
    population_size = 4
    keys = jax.random.split(key, population_size)

    # Create a batch of solutions
    solutions = jax.vmap(problem.sample)(keys)

    # Evaluate the solutions
    key_eval = jax.random.key(42)
    fitness, new_state, info = problem.eval(key_eval, solutions, state)

    # Check shape (population_size,)
    assert fitness.shape == (population_size,)
    assert new_state.counter == state.counter + 1
