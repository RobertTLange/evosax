"""Benchmark pretrained learned algorithms on a two-dimensional sphere."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from evosax.algorithms.distribution_based import EvoTF_ES, LearnedES
from evosax.algorithms.population_based import LearnedGA

NUM_DIMS = 2
POPULATION_SIZE = 8
ALGORITHMS = {
    "LES": LearnedES,
    "LGA": LearnedGA,
    "EvoTF": EvoTF_ES,
}


def run_benchmark(
    num_seeds: int = 10, num_generations: int = 100
) -> dict[str, np.ndarray]:
    """Collect best-so-far sphere fitness traces."""
    traces = {}
    for name, algorithm_class in ALGORITHMS.items():
        algorithm = algorithm_class(
            population_size=POPULATION_SIZE,
            solution=jnp.zeros(NUM_DIMS),
        )
        params = algorithm.default_params
        traces[name] = np.stack(
            [
                _run_algorithm(algorithm, params, seed, num_generations)
                for seed in range(num_seeds)
            ]
        )
    return traces


def plot_benchmark(traces: dict[str, np.ndarray], output_path: Path) -> None:
    """Plot median best fitness with an interquartile band."""
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for name, trace in traces.items():
        evaluations = POPULATION_SIZE * np.arange(1, trace.shape[1] + 1)
        median = np.median(trace, axis=0)
        lower, upper = np.percentile(trace, [25, 75], axis=0)
        (line,) = axis.plot(evaluations, median, linewidth=2, label=name)
        axis.fill_between(evaluations, lower, upper, color=line.get_color(), alpha=0.18)

    axis.set_yscale("log")
    axis.set_xlabel("Objective evaluations")
    axis.set_ylabel("Best sphere fitness (lower is better)")
    num_seeds = next(iter(traces.values())).shape[0]
    axis.set_title(
        f"Learned optimizers on 2-D sphere ({num_seeds} seeds; median + IQR)"
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _run_algorithm(algorithm, params, seed: int, num_generations: int) -> np.ndarray:
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    state, best_fitness = _initialize_state(algorithm, params, init_key)

    for _ in range(num_generations - len(best_fitness)):
        key, ask_key, tell_key = jax.random.split(key, 3)
        population, state = algorithm.ask(ask_key, state, params)
        fitness = jnp.sum(jnp.square(population), axis=-1)
        state, metrics = algorithm.tell(tell_key, population, fitness, state, params)
        _raise_for_nonfinite(population, fitness, metrics)
        best_fitness.append(metrics["best_fitness"])

    return np.asarray(jnp.asarray(best_fitness))


def _initialize_state(algorithm, params, key: jax.Array):
    mean = jnp.ones(NUM_DIMS)
    if isinstance(algorithm, LearnedGA):
        population = mean + jax.random.normal(key, (POPULATION_SIZE, NUM_DIMS))
        fitness = jnp.sum(jnp.square(population), axis=-1)
        state = algorithm.init(key, population, fitness, params)
        return state, [jnp.min(fitness)]

    state = algorithm.init(key, mean, params)
    return state, []


def _raise_for_nonfinite(population, fitness, metrics) -> None:
    values = (population, fitness, *jax.tree.leaves(metrics))
    if not all(jnp.all(jnp.isfinite(value)) for value in values):
        raise FloatingPointError("learned sphere benchmark produced non-finite values")


def main() -> None:
    """Generate the benchmark figure used in the compatibility pull request."""
    output_path = Path(__file__).parents[1] / "docs" / "figures" / "learned-sphere.png"
    traces = run_benchmark()
    plot_benchmark(traces, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
