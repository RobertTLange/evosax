import jax.numpy as jnp
from jax import vmap, jit


def rosenbrock_d_dim(x, a=1, b=100):
    '''
    D-Dim. Rosenbrock Fct. Evaluation.
    Value 0 - Minumum at x=a
    '''
    x_i, x_sq, x_p = x[:-1], x[:-1]**2, x[1:]
    return jnp.sum((a - x_i)**2 + b*(x_p-x_sq)**2)


def himmelblau_fct(x, offset=0):
    '''
    2-dim. Himmelblau function.
    Value 0 - Minima at [3, 2], [-2.81, 3.13],
                        [-3.78, -3.28], [3.58, -1.85]
    '''
    x = x + offset
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def six_hump_camel_fct(x, offset=0.):
    '''
    2-dim. 6-Hump Camel function.
    Value -1.0316 - Minimum at [0.0898, -0.7126], [-0.0898, 0.7126]
    '''
    x = x + offset
    p1 = (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2
    p2 = x[0] * x[1]
    p3 = (-4 + 4*x[1]**2)*x[1]**2
    return p1 + p2 + p3


# Toy Problem Evaluation Batch-Jitted Versions
batch_rosenbrock = jit(vmap(rosenbrock_d_dim, in_axes=(0, None, None),
                            out_axes=0))

batch_himmelblau = jit(vmap(himmelblau_fct, in_axes=(0, None),
                            out_axes=0))

batch_hump_camel = jit(vmap(six_hump_camel_fct, in_axes=(0, None),
                            out_axes=0))
