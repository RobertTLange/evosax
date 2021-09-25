import jax.numpy as jnp
from jax import vmap, jit


def quadratic_fitness(x):
    '''
    Simple 3-dim. quadratic function.
    f(x*)=0 - Minimum at [0.5, 0.1, -0.3]
    '''
    fit = jnp.sum(jnp.square(x - jnp.array([0.5, 0.1, -0.3])))
    return fit


def himmelblau_fct(x, offset=0):
    '''
    2-dim. Himmelblau function.
    f(x*)=0 - Minima at [3, 2], [-2.81, 3.13],
                        [-3.78, -3.28], [3.58, -1.85]
    '''
    x = x + offset
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def six_hump_camel_fct(x, offset=0.):
    '''
    2-dim. 6-Hump Camel function.
    f(x*)=-1.0316 - Minimum at [0.0898, -0.7126], [-0.0898, 0.7126]
    '''
    x = x + offset
    p1 = (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2
    p2 = x[0] * x[1]
    p3 = (-4 + 4*x[1]**2)*x[1]**2
    return p1 + p2 + p3


def rosenbrock_d_dim(x, a=1, b=100):
    '''
    D-Dim. Rosenbrock function. x_i ∈ [-32.768, 32.768] or x_i ∈ [-5, 10]
    f(x*)=0 - Minumum at x*=a
    '''
    x_i, x_sq, x_p = x[:-1], x[:-1]**2, x[1:]
    return jnp.sum((a - x_i)**2 + b*(x_p-x_sq)**2)


def ackley_d_dim(x, a=20, b=0.2, c=2*jnp.pi):
    '''
    D-Dim. Ackley function. x_i ∈ [-32.768, 32.768]
    f(x*)=0 - Minimum at x*=[0,...,0]
    '''
    return (- a * jnp.exp(-b * jnp.sqrt(jnp.mean(x**2)))
            - jnp.exp(jnp.mean(jnp.cos(c * x))) + a + jnp.exp(1))


def griewank_d_dim(x):
    '''
    D-Dim. Griewank function. x_i ∈ [-600, 600]
    f(x*)=0 - Minimum at x*=[0,...,0]
    '''
    return (jnp.sum(x**2/4000) -
            jnp.prod(jnp.cos(x/jnp.sqrt(jnp.arange(1, x.shape[0]+1)))) + 1)


def rastrigin_d_dim(x, A=10):
    '''
    D-Dim. Rastrigin function. x_i ∈ [-5.12, 5.12]
    f(x*)=0 - Minimum at x*=[0,...,0]
    '''
    return (A * x.shape[0] +
            jnp.sum(x**2 - A * jnp.cos(2* jnp.pi * x)))


def schwefel_d_dim(x):
    '''
    D-Dim. Schwefel function. x_i ∈ [-500, 500]
    f(x*)=0 - Minimum at x*=[420.9687,...,420.9687]
    '''
    return (418.9829 * x.shape[0] -
            jnp.sum(x * jnp.sin(jnp.sqrt(jnp.absolute(x)))))


# Toy Problem Evaluation Batch-Jitted Versions
batch_quadratic = jit(vmap(quadratic_fitness, 0))

batch_himmelblau = jit(vmap(himmelblau_fct, in_axes=(0, None),
                            out_axes=0))

batch_hump_camel = jit(vmap(six_hump_camel_fct, in_axes=(0, None),
                            out_axes=0))

batch_rosenbrock = jit(vmap(rosenbrock_d_dim, in_axes=(0, None, None),
                            out_axes=0))

batch_ackley = jit(vmap(ackley_d_dim, in_axes=(0, None, None, None),
                   out_axes=0))

batch_griewank = jit(vmap(griewank_d_dim, in_axes=(0,), out_axes=0))

batch_rastrigin = jit(vmap(rastrigin_d_dim, in_axes=(0, None), out_axes=0))

batch_schwefel = jit(vmap(schwefel_d_dim, in_axes=(0,), out_axes=0))
