"""Protocol and restart checks without expensive physics compilations."""

from pathlib import Path
from runpy import run_path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evosax.problems.rl.brax import State as BraxState
from flax import struct

EXAMPLE = run_path(Path(__file__).parents[2] / "examples/cma_anchored_les_brax.py")
Config = EXAMPLE["Config"]
run_seed = EXAMPLE["run_seed"]


@struct.dataclass
class FakeParams:
    std_init: float = 0.05


class FakeStrategy:
    default_params = FakeParams()

    def init(self, key, mean, params):
        return mean

    def ask(self, key, state, params):
        return state + jax.random.normal(key, (4,)) * params.std_init, state + 1

    def tell(self, key, population, fitness, state, params):
        return population[jnp.nanargmin(fitness)], {}


class FakeProblem:
    def sample(self, key):
        return jnp.array(0.0)

    def init(self, key):
        return jnp.array(0)

    def eval(self, key, population, state):
        return population + jax.random.uniform(key) + state, state + 1, {}


def test_defaults_match_broader_transfer_protocol():
    config = Config()
    assert len(config.environments) == 8
    assert config.seeds == tuple(range(20000, 20016))
    assert (config.population_size, config.generations, config.episode_length) == (
        256,
        2000,
        500,
    )
    assert config.sigma_init == 0.05


def test_resume_reproduces_uninterrupted_curves_and_state(tmp_path):
    config = Config(generations=5, population_size=4)
    args = (FakeStrategy(), FakeProblem(), config, 20000)
    identity = {"runtime": "test"}
    complete, _ = run_seed(*args, tmp_path / "full.msgpack", identity)
    run_seed(*args, tmp_path / "split.msgpack", identity, stop_after=2)
    resumed, _ = run_seed(*args, tmp_path / "split.msgpack", identity, resume=True)
    np.testing.assert_array_equal(complete, resumed)
    assert np.all(np.diff(resumed) >= 0)
    load = EXAMPLE["serialization"].msgpack_restore
    full_state = load((tmp_path / "full.msgpack").read_bytes())["state"]
    split_state = load((tmp_path / "split.msgpack").read_bytes())["state"]
    for actual, expected in zip(
        jax.tree.leaves(full_state), jax.tree.leaves(split_state)
    ):
        np.testing.assert_array_equal(actual, expected)


def test_resume_rejects_mismatch_and_existing_outputs(tmp_path):
    args = (FakeStrategy(), FakeProblem(), Config(generations=1), 20000)
    checkpoint = tmp_path / "state.msgpack"
    run_seed(*args, checkpoint, {"runtime": "first"})
    with pytest.raises(FileExistsError):
        run_seed(*args, checkpoint, {"runtime": "first"})
    with pytest.raises(ValueError, match="identity"):
        run_seed(*args, checkpoint, {"runtime": "changed"}, resume=True)


def test_validity_ignores_post_terminal_observations_but_rejects_bad_rollouts():
    observations = jnp.array([[[[1.0], [2.0], [jnp.nan]]], [[[1.0], [jnp.inf], [0.0]]]])
    valid, mask = EXAMPLE["rollout_validity"](
        jnp.array([[3.0], [4.0]]), observations, jnp.array([[[0, 1, 0]], [[0, 0, 0]]])
    )
    np.testing.assert_array_equal(valid, [[True], [False]])
    np.testing.assert_array_equal(
        mask, [[[True, True, False]], [[False, False, False]]]
    )


def test_masked_normalization_excludes_invalid_and_empty_batches():
    state = BraxState(
        counter=0,
        obs_mean=jnp.zeros(1),
        obs_std=jnp.ones(1),
        obs_var_sum=jnp.zeros(1),
        obs_counter=0,
        std_min=1e-6,
        std_max=1e6,
    )
    update = EXAMPLE["update_masked_stats"]
    observations = jnp.array([[[[2.0], [4.0], [jnp.nan]]]])
    updated = update(observations, jnp.array([[[True, True, False]]]), state)
    np.testing.assert_allclose(updated.obs_mean, [3.0])
    np.testing.assert_allclose(updated.obs_std, [1.0])
    assert updated.obs_counter == 2
    empty = update(observations, jnp.zeros((1, 1, 3), dtype=bool), updated)
    np.testing.assert_array_equal(empty.obs_mean, updated.obs_mean)
    assert empty.obs_counter == 2


def test_tell_negates_rewards_and_skips_all_invalid_generations():
    stages = EXAMPLE["make_stages"](FakeStrategy(), FakeProblem())
    State = EXAMPLE["SeedState"]
    state = State(jnp.array(2.0), jnp.array(0), jnp.array(-1000001.0))
    key = jax.random.key(0)
    population = jnp.array([1.0, 2.0, 3.0, 4.0])
    updated = stages[2](
        state,
        state.strategy_state + 1,
        population,
        jnp.array([1.0, jnp.nan, 3.0, 2.0]),
        state.problem_state,
        key,
        FakeParams(),
    )
    assert updated.strategy_state == 3.0
    assert updated.best_return == 3.0
    invalid = stages[2](
        updated,
        updated.strategy_state + 1,
        population,
        jnp.full(4, jnp.nan),
        state.problem_state,
        key,
        FakeParams(),
    )
    assert invalid.strategy_state == updated.strategy_state
    assert invalid.best_return == updated.best_return


@pytest.mark.parametrize(
    "arguments",
    [
        ["--generations", "0"],
        ["--sigma-init", "nan"],
        ["--seeds", "-1"],
        ["--environments", "invalid"],
    ],
)
def test_cli_rejects_invalid_configuration(arguments):
    with pytest.raises(SystemExit):
        EXAMPLE["parse_args"](arguments)


def test_plot_and_summary_write_finite_results(tmp_path):
    curves = {"ant": [np.array([1.0, 3.0]), np.array([2.0, 4.0])]}
    EXAMPLE["write_summary"](curves, tmp_path)
    import json

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["ant"]["final_return_mean"] == 3.5
    assert summary["ant"]["auc_mean"] == 2.5
    assert (tmp_path / "learning_curves.png").stat().st_size > 0


def test_legacy_problem_zero_policy_and_shared_rollout_randomness(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    @struct.dataclass
    class EnvState:
        obs: jax.Array
        reward: jax.Array
        done: jax.Array

    class Environment:
        action_size = 1

        def reset(self, key):
            return EnvState(
                jax.random.uniform(key, (1,)), jnp.array(0.0), jnp.array(0.0)
            )

        def step(self, state, action):
            return state.replace(obs=state.obs + 1, reward=state.obs[0] + action[0])

    created = []

    def create(**kwargs):
        created.append(kwargs)
        return Environment()

    v1 = ModuleType("brax.v1")
    v1.envs = SimpleNamespace(create=create)
    monkeypatch.setitem(sys.modules, "brax.v1", v1)
    problem = EXAMPLE["LegacySpringProblem"]("reacher", 3)
    assert created[0]["legacy_spring"] is True
    key = jax.random.key(0)
    solution = problem.sample(key)
    assert all(np.all(np.asarray(leaf) == 0) for leaf in jax.tree.leaves(solution))
    population = jax.tree.map(lambda x: jnp.stack([x, x]), solution)
    returns, state, _ = problem.eval(key, population, problem.init(key))
    assert returns[0] == returns[1]
    assert state.obs_counter == 6
    with pytest.raises(ValueError):
        EXAMPLE["LegacySpringProblem"]("unsupported", 3)


def test_runtime_identity_includes_distinct_sources_and_weights():
    strategy = EXAMPLE["CMAAnchoredLES"](4, jnp.zeros(2))
    identity = EXAMPLE["runtime_identity"](strategy)
    assert "algorithms/base.py" in identity["sources"]
    assert all("\\" not in name for name in identity["sources"])
    assert "algorithms/distribution_based/base.py" in identity["sources"]
    assert len(identity["weights_sha256"]) == 64
    assert identity["packages"]["jax"] == jax.__version__


def test_cell_identity_allows_subsets_but_rejects_protocol_changes():
    identity = EXAMPLE["cell_identity"]
    original = identity(Config(), "ant", 20000, {})
    subset = Config(environments=("ant",), seeds=(20000,))
    assert identity(subset, "ant", 20000, {}) == original
    assert identity(Config(generations=1), "ant", 20000, {}) != original


def test_main_writes_metadata_and_resumes(monkeypatch, tmp_path):
    import json

    main = EXAMPLE["main"]
    monkeypatch.setitem(
        main.__globals__, "LegacySpringProblem", lambda *args: FakeProblem()
    )
    monkeypatch.setitem(
        main.__globals__, "CMAAnchoredLES", lambda *args: FakeStrategy()
    )
    monkeypatch.setitem(
        main.__globals__, "runtime_identity", lambda strategy: {"runtime": "fake"}
    )
    arguments = [
        "--environments",
        "ant",
        "--seeds",
        "20000",
        "--generations",
        "2",
        "--population-size",
        "4",
        "--output-dir",
        str(tmp_path),
    ]
    main(arguments)
    main(arguments + ["--resume"])
    results = json.loads((tmp_path / "ant/20000/results.json").read_text())
    assert len(results["best_returns"]) == 2
    assert results["identity"]["config"]["generations"] == 2
    assert results["runtime_seconds"] >= 0

    main(
        [
            "--environments",
            "reacher",
            "--seeds",
            "20000",
            "20001",
            "--generations",
            "2",
            "--population-size",
            "4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    main(
        [
            "--environments",
            "reacher",
            "--seeds",
            "20000",
            "--generations",
            "2",
            "--population-size",
            "4",
            "--output-dir",
            str(tmp_path),
            "--resume",
        ]
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["ant"]["seeds"] == 1
    assert summary["reacher"]["seeds"] == 2


def test_mixed_invalid_returns_use_finite_penalty():
    class RecordingStrategy(FakeStrategy):
        def tell(self, key, population, fitness, state, params):
            return fitness, {}

    tell = EXAMPLE["make_stages"](RecordingStrategy(), FakeProblem())[2]
    state = EXAMPLE["SeedState"](jnp.zeros(4), jnp.array(0), jnp.array(-1000001.0))
    updated = tell(
        state,
        state.strategy_state,
        jnp.zeros(4),
        jnp.array([3.0, jnp.nan, jnp.inf, -1000001.0]),
        state.problem_state,
        jax.random.key(0),
        FakeParams(),
    )
    np.testing.assert_array_equal(
        updated.strategy_state, [-3.0, 1000001.0, 1000001.0, 1000001.0]
    )
