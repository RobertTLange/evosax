class Strategy(object):
    def __init__(self, num_dims: int, popsize: int):
        """Base Abstract Class for an Evolutionary Strategy."""
        self.num_dims = num_dims
        self.popsize = popsize

    def default_params(self):
        """Return default parameters of evolutionary strategy."""
        raise NotImplementedError

    def initialize(self, rng, params):
        """`initialize` the evolutionary strategy."""
        raise NotImplementedError

    def ask(self, rng, state, params):
        """`ask` for new parameter candidates to evaluate next."""
        raise NotImplementedError

    def tell(self, x, fitness, state, params):
        """`tell` performance data for strategy state update."""
        raise NotImplementedError
