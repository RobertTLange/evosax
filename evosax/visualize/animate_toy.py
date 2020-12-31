import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pylab import rcParams

from evosax.problems.low_d_optimisation import (batch_hump_camel,
                                                batch_himmelblau,
                                                batch_rosenbrock)

fct_name = "rosenbrock"
file_name = "rosenbrock.gif"
rng = jax.random.PRNGKey(0)

rcParams["figure.figsize"] = 5, 10
fig, (ax1, ax2) = plt.subplots(2, 1)

color_dict = {
    "red": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "blue": ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "yellow": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
}
bw = LinearSegmentedColormap("BlueWhile", color_dict)


def six_hump_camel_contour(x):
    return np.log(six_hump_camel_fct(x) + 1.0316)

def himmelblau_contour(x):
    return np.log(himmelblau_fct(x) + 1)

def rosenbrock_fct(x):
    x_i, x_sq, x_p = x[0], x[0]**2, x[1]
    return (1 - x_i)**2 + 100*(x_p-x_sq)**2

def rosenbrock_contour(x):
    return np.log(rosenbrock_fct(x) + 1)

if fct_name == "six-hump":
    function_name = "Six-Hump Camel Fct."
    objective = batch_hump_camel
    contour_function = six_hump_camel_contour
    global_minimums = [(0.0898, -0.7126),
                       (-0.0898, 0.7126),]
    x1_lower_bound, x1_upper_bound = -3, 3
    x2_lower_bound, x2_upper_bound = -2, 2
elif fct_name == "himmelblau":
    function_name = "Himmelblau Fct."
    objective = batch_himmelblau
    contour_function = himmelblau_contour
    global_minimums = [(3.0, 2.0),
                       (-2.805118, 3.131312),
                       (-3.779310, -3.283186),
                       (3.584428, -1.848126),]
    x1_lower_bound, x1_upper_bound = -4, 4
    x2_lower_bound, x2_upper_bound = -4, 4
elif fct_name == "rosenbrock":
    function_name = "Rosenbrock Fct."
    objective = batch_rosenbrock
    contour_function = rosenbrock_contour
    global_minimums = [(1, 1),]
    x1_lower_bound, x1_upper_bound = -5, 10
    x2_lower_bound, x2_upper_bound = -5, 10


sigma = (x1_upper_bound - x2_lower_bound) / 5
es_params, es_memory = init_cma_es(jnp.zeros(2), sigma,
                                   population_size=4, mu=2)
evo_logger = init_evo_logger(4, 2)


def init():
    ax1.set_xlim(x1_lower_bound, x1_upper_bound)
    ax1.set_ylim(x2_lower_bound, x2_upper_bound)
    ax2.set_xlim(x1_lower_bound, x1_upper_bound)
    ax2.set_ylim(x2_lower_bound, x2_upper_bound)
    ax1.axis("off")
    ax2.axis("off")

    # Plot 4 local minimum value
    for m in global_minimums:
        ax1.plot(m[0], m[1], "y*", ms=10)
        ax2.plot(m[0], m[1], "y*", ms=10)

    # Plot contour of himmelbleu function
    x1 = np.arange(x1_lower_bound, x1_upper_bound, 0.01)
    x2 = np.arange(x2_lower_bound, x2_upper_bound, 0.01)
    x1, x2 = np.meshgrid(x1, x2)
    x = np.stack([x1, x2])
    ax1.contour(x1, x2, contour_function(x), 30, cmap=bw)
    fig.tight_layout()


def update(frame):
    global es_params, es_memory, evo_logger, rng

    rng, rng_input = jax.random.split(rng)
    x, es_memory = ask_cma_es(rng_input, es_memory, es_params)
    value = objective(x, 1, 100)
    es_memory = tell_cma_es(x, value, 2, es_params, es_memory)

    for i in range(4):
        # Plot sample points
        ax1.plot(x[i, 0], x[i, 1], "o", c="r", alpha=0.5)

    #fig.subplots_adjust(top=0.95)
    #fig.suptitle(f"{function_name}", fontsize=25, y=1.05)

    # Plot multivariate gaussian distribution of CMA-ES
    x, y = np.mgrid[x1_lower_bound:x1_upper_bound:0.01,
                    x2_lower_bound:x2_upper_bound:0.01]
    rv = stats.multivariate_normal(np.array(es_memory["mean"]),
                                   np.array(es_memory["C"]))
    pos = np.dstack((x, y))
    ax2.contourf(x, y, rv.pdf(pos), cmap="Reds")

    if frame % 10 == 0:
        print(f"Processing frame {frame}")


def main():
    ani = animation.FuncAnimation(fig, update, frames=50,
                                  init_func=init, blit=False,
                                  interval=100)
    ani.save(file_name, dpi=300, writer='imagemagick')


if __name__ == "__main__":
    main()
