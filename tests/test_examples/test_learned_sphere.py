"""Tests for the learned-algorithm sphere benchmark."""

from pathlib import Path
from runpy import run_path

import numpy as np

EXAMPLE_PATH = Path(__file__).parents[2] / "examples" / "learned_sphere.py"
EXAMPLE = run_path(EXAMPLE_PATH)
plot_benchmark = EXAMPLE["plot_benchmark"]
run_benchmark = EXAMPLE["run_benchmark"]


def test_benchmark_returns_finite_improving_traces():
    """Every learned algorithm must improve during the benchmark."""
    traces = run_benchmark(num_seeds=1, num_generations=32)

    for trace in traces.values():
        assert trace.shape == (1, 32)
        assert np.all(np.isfinite(trace))
        assert trace[0, -1] < trace[0, 0]


def test_plot_benchmark_writes_figure(tmp_path):
    """Benchmark traces can be rendered for reports and pull requests."""
    traces = {
        "Example": np.array(
            [
                [2.0, 1.0, 0.5],
                [2.0, 0.8, 0.4],
            ]
        )
    }
    output_path = tmp_path / "sphere.png"

    plot_benchmark(traces, output_path)

    assert output_path.stat().st_size > 0
