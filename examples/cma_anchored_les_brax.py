"""Broader eight-task Brax transfer evaluation with frozen CMAAnchoredLES weights.

Run from an editable evosax checkout (``pip install -e .``). Historical experiment:
Python 3.10, jax[cuda12]==0.6.2, flax==0.10.7, brax==0.12.3,
chex==0.1.90, mujoco==3.3.3, numpy==2.2.6, optax==0.2.5,
orbax-checkpoint==0.10.3, scipy==1.15.3. Use jax==0.6.2 for CPU.
These are reproduction pins, not a recommendation for new projects.

Smoke run::

    python examples/cma_anchored_les_brax.py --environments reacher --seeds 20000 \
        --generations 2 --population-size 4 --episode-length 8 \
        --output-dir /tmp/brax-smoke

Omit overrides for the full protocol; add --resume to continue a matching run.
Each environment/seed runs sequentially on one device. Independent launches can
select disjoint environments/seeds and should use separate output directories.
The historical harness used eight devices; execution layout is the only intended
protocol difference. Floating-point results can differ across devices/runtimes.
Plots show mean best return +/- one population standard deviation across seeds;
AUC is the mean best return over generations, as in the transfer analysis.
No comparator-normalized scores or retraining are performed.
"""

import argparse
import hashlib
import inspect
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import CMAAnchoredLES
from evosax.problems import BraxProblem
from flax import linen as nn, serialization

ENVIRONMENTS = (
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "reacher",
    "walker2d",
    "fetch",
    "ur5e",
)
MAX_ABS_VALUE = 1_000_000.0
INVALID_RETURN = -1_000_001.0
CHECKPOINT_INTERVAL = 250


@dataclass(frozen=True)
class Config:
    """Defaults of the broader transfer experiment."""

    environments: tuple[str, ...] = ENVIRONMENTS
    seeds: tuple[int, ...] = tuple(range(20_000, 20_016))
    generations: int = 2_000
    population_size: int = 256
    episode_length: int = 500
    sigma_init: float = 0.05


class LinearPolicy(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, observation, key):
        del key
        return jnp.tanh(nn.Dense(self.action_size)(observation))


def rollout_validity(fitness, observations, done):
    """Reject invalid active trajectories; exclude steps after termination."""
    continuation = jnp.where(jnp.isfinite(done), 1.0 - done, 0.0)
    active = jnp.concatenate(
        (
            jnp.ones(done.shape[:-1] + (1,), dtype=done.dtype),
            jnp.cumprod(continuation[..., :-1], axis=-1),
        ),
        axis=-1,
    ).astype(jnp.bool_)
    finite_return = jnp.isfinite(fitness) & (jnp.abs(fitness) <= MAX_ABS_VALUE)
    finite_observation = jnp.all(
        jnp.isfinite(observations) & (jnp.abs(observations) <= MAX_ABS_VALUE), axis=-1
    )
    valid = finite_return & jnp.all((~active) | finite_observation, axis=-1)
    return valid, active & valid[..., None]


def update_masked_stats(observations, validity, state):
    """Welford update using only active observations from valid rollouts."""
    safe_observations = jnp.where(validity[..., None], observations, state.obs_mean)
    count = jnp.sum(validity, dtype=jnp.asarray(state.obs_counter).dtype)
    new_counter = state.obs_counter + count
    denominator = jnp.maximum(new_counter, 1)
    old_delta = safe_observations - state.obs_mean
    mean = state.obs_mean + jnp.sum(old_delta, axis=(0, 1, 2)) / denominator
    var_sum = jnp.maximum(
        state.obs_var_sum
        + jnp.sum(old_delta * (safe_observations - mean), axis=(0, 1, 2)),
        0,
    )
    std = jnp.clip(jnp.sqrt(var_sum / denominator), state.std_min, state.std_max)
    return state.replace(
        obs_mean=jnp.where(count > 0, mean, state.obs_mean),
        obs_std=jnp.where(count > 0, std, state.obs_std),
        obs_var_sum=jnp.where(count > 0, var_sum, state.obs_var_sum),
        obs_counter=jnp.where(count > 0, new_counter, state.obs_counter),
    )


class LegacySpringProblem(BraxProblem):
    """Historical Brax v1 physics with zero affine+tanh policies."""

    def __init__(self, environment, episode_length):
        from brax.v1 import envs

        if environment not in ENVIRONMENTS or episode_length <= 0:
            raise ValueError("unsupported environment or nonpositive episode length")
        self.env_name = environment
        self.episode_length = episode_length
        self.num_rollouts = 1
        self.use_normalize_obs = True
        self.env = envs.create(
            env_name=environment,
            episode_length=episode_length,
            auto_reset=True,
            legacy_spring=True,
        )
        self.policy = LinearPolicy(self.env.action_size)
        key = jax.random.key(0)
        env_state = self.env.reset(key)
        policy_params = self.policy.init(key, env_state.obs, key)
        self.env.step(env_state, self.policy.apply(policy_params, env_state.obs, key))
        self._rollouts = jax.vmap(self._rollout, in_axes=(0, None, None))
        self._eval = jax.vmap(self._rollouts, in_axes=(None, 0, None))

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key):
        key_init, key_reset, key_input = jax.random.split(key, 3)
        params = self.policy.init(key_init, self.env.reset(key_reset).obs, key_input)
        return jax.tree.map(jnp.zeros_like, params)

    @partial(jax.jit, static_argnames=("self",))
    def eval(self, key, solutions, state):
        keys = jax.random.split(key, self.num_rollouts)
        fitness, env_states = self._eval(keys, solutions, state)
        valid, mask = rollout_validity(fitness, env_states.obs, env_states.done)
        state = update_masked_stats(env_states.obs, mask, state)
        fitness = jnp.where(valid, fitness, jnp.nan)
        return jnp.mean(fitness, axis=-1), state.replace(counter=state.counter + 1), {}


class SeedState(NamedTuple):
    strategy_state: Any
    problem_state: Any
    best_return: jax.Array


def make_stages(strategy, problem):
    """Keep optimizer and physics compilations separate."""

    @jax.jit
    def ask(state, base_key, generation, params):
        generation_key = jax.random.fold_in(base_key, generation + 1)
        ask_key, eval_key, tell_key = jax.random.split(generation_key, 3)
        population, asked = strategy.ask(ask_key, state, params)
        return population, asked, eval_key, tell_key

    @jax.jit
    def evaluate(key, population, state):
        returns, state, _ = problem.eval(key, population, state)
        return returns, state

    @jax.jit
    def tell(state, asked, population, returns, problem_state, key, params):
        valid = jnp.isfinite(returns) & (jnp.abs(returns) <= MAX_ABS_VALUE)
        next_strategy = jax.lax.cond(
            jnp.any(valid),
            lambda: strategy.tell(
                key,
                population,
                -jnp.where(valid, returns, INVALID_RETURN),
                asked,
                params,
            )[0],
            lambda: state.strategy_state,
        )
        best = jnp.maximum(
            state.best_return, jnp.max(jnp.where(valid, returns, INVALID_RETURN))
        )
        return SeedState(next_strategy, problem_state, best)

    return ask, evaluate, tell


def atomic_write(path, data):
    """Replace complete artifacts atomically; never deserialize executable objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def run_seed(
    strategy,
    problem,
    config,
    seed,
    checkpoint,
    identity,
    resume=False,
    stop_after=None,
    stages=None,
):
    """Run/resume one seed, persisting optimizer, normalization, and best curves."""
    if checkpoint.exists() and not resume:
        raise FileExistsError(
            f"{checkpoint} exists; use --resume or a new output directory"
        )
    params = strategy.default_params.replace(std_init=config.sigma_init)
    key = jax.random.key(seed)
    mean_key, problem_key, strategy_key = jax.random.split(key, 3)
    state = SeedState(
        strategy.init(strategy_key, problem.sample(mean_key), params),
        problem.init(problem_key),
        jnp.asarray(INVALID_RETURN, jnp.float32),
    )
    curves, prior_runtime = [], 0.0
    if checkpoint.exists():
        saved = serialization.msgpack_restore(checkpoint.read_bytes())
        if saved["identity"] != identity:
            raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
        curves = np.asarray(saved["curves"], dtype=np.float32).tolist()
        if (
            len(curves) != saved["next_generation"]
            or len(curves) > config.generations
            or not np.all(np.isfinite(curves))
        ):
            raise ValueError(f"invalid checkpoint progress: {checkpoint}")
        state = serialization.from_state_dict(state, saved["state"])
        state = jax.tree.map(jnp.asarray, state)
        prior_runtime = float(saved["runtime_seconds"])
    stop = config.generations if stop_after is None else stop_after
    if not len(curves) <= stop <= config.generations:
        raise ValueError("stop generation outside checkpoint/configuration range")
    stages = make_stages(strategy, problem) if stages is None else stages
    started = time.perf_counter()
    for generation in range(len(curves), stop):
        population, asked, eval_key, tell_key = stages[0](
            state.strategy_state, key, jnp.asarray(generation, jnp.uint32), params
        )
        returns, problem_state = stages[1](eval_key, population, state.problem_state)
        state = stages[2](
            state, asked, population, returns, problem_state, tell_key, params
        )
        curves.append(float(state.best_return))
        if (generation + 1) % CHECKPOINT_INTERVAL == 0 or generation + 1 == stop:
            runtime = prior_runtime + time.perf_counter() - started
            saved = dict(
                identity=identity,
                next_generation=generation + 1,
                state=serialization.to_state_dict(jax.device_get(state)),
                curves=np.asarray(curves, np.float32),
                runtime_seconds=runtime,
            )
            atomic_write(checkpoint, serialization.msgpack_serialize(saved))
            print(
                f"seed={seed} generations={generation + 1}/{config.generations} "
                f"best={curves[-1]:.3f} elapsed={runtime:.1f}s",
                flush=True,
            )
    return np.asarray(curves, np.float32), prior_runtime + time.perf_counter() - started


def runtime_identity(strategy):
    """Capture software, device, source, and frozen parameter identity."""
    packages = {}
    for name in (
        "evosax",
        "jax",
        "jaxlib",
        "flax",
        "brax",
        "numpy",
        "scipy",
        "mujoco",
        "chex",
        "optax",
        "orbax-checkpoint",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    import evosax

    root = Path(evosax.__file__).parent
    sources = [
        Path(__file__),
        Path(inspect.getfile(type(strategy))),
        root / "problems/rl/brax.py",
        root / "algorithms/base.py",
        root / "algorithms/distribution_based/base.py",
        root / "core/fitness_shaping.py",
    ]
    sources += sorted((root / "learned_evolution").glob("cma_anchored*.py"))
    hashes = {
        str(path.relative_to(root))
        if path.is_relative_to(root)
        else path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    params = serialization.to_state_dict(jax.device_get(strategy.default_params.params))
    weights = hashlib.sha256(serialization.msgpack_serialize(params)).hexdigest()
    return dict(
        packages=packages,
        python=platform.python_version(),
        backend=jax.default_backend(),
        device=jax.devices()[0].device_kind,
        x64=jax.config.jax_enable_x64,
        prng=str(jax.config.jax_default_prng_impl),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        sources=hashes,
        weights_sha256=weights,
    )


def cell_identity(config, environment, seed, runtime):
    settings = asdict(config)
    settings.pop("environments")
    settings.pop("seeds")
    return dict(
        format_version=1,
        environment=environment,
        seed=seed,
        config=settings,
        runtime=runtime,
    )


def write_summary(results, output):
    """Aggregate selected seeds and render all eight task panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for environment, axis in zip(ENVIRONMENTS, axes.flat):
        axis.set_title(environment)
        axis.set_xlabel("Generation")
        axis.set_ylabel("Best return")
        if environment not in results:
            axis.text(0.5, 0.5, "Not selected", ha="center", transform=axis.transAxes)
            continue
        values = np.stack(results[environment])
        mean, std = values.mean(axis=0), values.std(axis=0)
        generation = np.arange(1, values.shape[1] + 1)
        axis.plot(generation, mean)
        axis.fill_between(generation, mean - std, mean + std, alpha=0.2)
        summary[environment] = dict(
            seeds=len(values),
            final_return_mean=float(values[:, -1].mean()),
            final_return_std=float(values[:, -1].std()),
            auc_mean=float(values.mean(axis=1).mean()),
            auc_std=float(values.mean(axis=1).std()),
        )
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    write_json(output / "summary.json", summary)


def completed_results(output, config, runtime):
    """Include compatible completed cells from earlier subset invocations."""
    results = {}
    for environment in ENVIRONMENTS:
        for path in sorted((output / environment).glob("*/results.json")):
            record = json.loads(path.read_text())
            seed = int(path.parent.name)
            if record["identity"] != cell_identity(config, environment, seed, runtime):
                continue
            curves = np.asarray(record["best_returns"], dtype=np.float32)
            if curves.shape != (config.generations,) or not np.all(np.isfinite(curves)):
                raise ValueError(f"invalid completed curves: {path}")
            results.setdefault(environment, []).append(curves)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--environments", nargs="+", choices=ENVIRONMENTS, default=ENVIRONMENTS
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=Config.seeds)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--sigma-init", "--std-init", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        type=Path,
        default=Path("brax_cma_anchored_les"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.generations, args.population_size, args.episode_length) <= 0:
        parser.error(
            "generations, population size, and episode length must be positive"
        )
    if not np.isfinite(args.sigma_init) or args.sigma_init <= 0:
        parser.error("sigma-init must be positive and finite")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error("seeds must be unsigned 32-bit integers")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.environments)) != len(
        args.environments
    ):
        parser.error("duplicate seeds/environments are not supported")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = Config(
        tuple(args.environments),
        tuple(args.seeds),
        args.generations,
        args.population_size,
        args.episode_length,
        args.sigma_init,
    )
    for environment in config.environments:
        print(f"Constructing {environment} (Brax v1 legacy_spring)", flush=True)
        problem = LegacySpringProblem(environment, config.episode_length)
        strategy = CMAAnchoredLES(
            config.population_size, problem.sample(jax.random.key(0))
        )
        runtime = runtime_identity(strategy)
        stages = make_stages(strategy, problem)
        for seed in config.seeds:
            identity = cell_identity(config, environment, seed, runtime)
            directory = args.output / environment / str(seed)
            curves, elapsed = run_seed(
                strategy,
                problem,
                config,
                seed,
                directory / "checkpoint.msgpack",
                identity,
                resume=args.resume,
                stages=stages,
            )
            write_json(
                directory / "results.json",
                dict(
                    identity=identity,
                    best_returns=curves.tolist(),
                    final_return=float(curves[-1]),
                    auc=float(curves.mean()),
                    runtime_seconds=elapsed,
                ),
            )
        write_summary(completed_results(args.output, config, runtime), args.output)


if __name__ == "__main__":
    main()
