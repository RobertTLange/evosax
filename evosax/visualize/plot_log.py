import matplotlib.pyplot as plt


def plot_fitness(evo_logger, title, ylims=(0, 10), fig=None, ax=None, no_legend=False):
    """Plot fitness trajectory from evo logger over generations."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(evo_logger["log_gen"], evo_logger["log_top_1"], label="Top 1")
    ax.plot(evo_logger["log_gen"], evo_logger["log_top_mean"], label="Top-k Mean")
    ax.plot(evo_logger["log_gen"], evo_logger["log_gen_1"], label="Gen. 1")
    ax.plot(evo_logger["log_gen"], evo_logger["log_gen_mean"], label="Gen. Mean")
    ax.set_ylim(ylims)
    if not no_legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Number of Generations")
    ax.set_ylabel("Fitness Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_sigma(evo_logger, title, ylims=(0, 1.5), fig=None, ax=None):
    """Plot sigma trace from evo logger over generations."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(evo_logger["log_gen"], evo_logger["log_sigma"])
    ax.set_ylim(ylims)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Number of Generations")
    ax.set_ylabel(r"Stepsize: $\sigma$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax
