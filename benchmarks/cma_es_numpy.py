import numpy as np


class CMA:
    """
    Minimal numpy CMA-ES with ask-and-tell user interface. Adapted from:
    https://github.com/CyberAgent/cmaes/blob/main/cmaes/_cma.py
    """
    def __init__(self, mean, sigma, seed, population_size=None, elite_size=None):
        self.n_dim = len(mean)
        self.mean = mean
        self.sigma = sigma
        self.generation = 0
        self._rng = np.random.RandomState(seed)
        self._EPS = 1e-20

        if population_size is None:
            self.population_size = 4 + np.floor(3 * np.log(self.n_dim))
        else:
            self.population_size = population_size
        if elite_size is None: self.mu = self.population_size // 2
        else: self.mu = elite_size

        self.init_strategy()
        check_cma_es_params(self.population_size, self.n_dim, self.sigma,
                            self.c_1, self.c_mu, self.c_sigma, self.c_c)

    def init_strategy(self):
        ''' Initialize evolutionary strategy & learning rates. '''
        # Weights for elite members
        weights_prime = np.array(
            [np.log((self.population_size + 1) / 2) - np.log(i + 1)
             for i in range(self.population_size)])
        self.mu_eff = ((np.sum(weights_prime[:self.mu]) ** 2) /
                  np.sum(weights_prime[:self.mu] ** 2))
        mu_eff_minus = ((np.sum(weights_prime[self.mu:]) ** 2) /
                        np.sum(weights_prime[self.mu:] ** 2))

        # lrates for rank-one and rank-μ C updates
        alpha_cov = 2
        self.c_1 = alpha_cov / ((self.n_dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1 - 1e-8, alpha_cov * (self.mu_eff - 2 + 1 / self.mu_eff)
                  / ((self.n_dim + 2) ** 2 + alpha_cov * self.mu_eff / 2))
        min_alpha = min(1 + self.c_1 / self.c_mu,
                        1 + (2 * mu_eff_minus) / (self.mu_eff + 2),
                        (1 - self.c_1 - self.c_mu) / (self.n_dim * self.c_mu))

        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        self.weights = np.where(weights_prime >= 0,
                           1 / positive_sum * weights_prime,
                           min_alpha / negative_sum * weights_prime,)
        self.c_m = 1

        # lrate for cumulation of step-size control and rank-one update
        self.c_sigma = (self.mu_eff + 2) / (self.n_dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n_dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.n_dim) / (self.n_dim + 4 + 2 * self.mu_eff / self.n_dim)
        self.chi_n = np.sqrt(self.n_dim) * (
            1.0 - (1.0 / (4.0 * self.n_dim)) + 1.0 / (21.0 * (self.n_dim ** 2)))

        # Initialize evolution paths & covariance matrix
        self.p_sigma = np.zeros(self.n_dim)
        self.p_c = np.zeros(self.n_dim)
        self._C, self._D, self._B = np.eye(self.n_dim), None, None

    def ask(self):
        """ Propose parameters to evaluate next. """
        B, D = self._eigen_decomposition()
        z = self._rng.randn(self.n_dim, self.population_size)  # ~ N(0, I)
        y = B.dot(np.diag(D)).dot(z)      # ~ N(0, C)
        y = np.swapaxes(y, 1, 0)
        x = self.mean + self.sigma * y  # ~ N(m, σ^2 C)
        return x

    def _eigen_decomposition(self):
        """ Perform eigendecomposition of covariance matrix. """
        if self._B is not None and self._D is not None:
            return self._B, self._D
        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, self._EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)
        self._B, self._D = B, D
        return B, D

    def tell(self, x, fitness):
        """ Update the surrogate ES model. """
        self.generation += 1
        # Sort new results, extract elite, store best performer
        concat_p_f = np.hstack([np.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        y_k, y_w, self.mean = self.update_mean(sorted_solutions)
        self.p_sigma, C_2 = self.update_p_sigma(y_w)
        self.p_c, norm_p_sigma, h_sigma = self.update_p_c(y_w)
        self._C = self.update_covariance(y_k, h_sigma, C_2)
        self.sigma = self.update_sigma(norm_p_sigma)

    def update_mean(self, sorted_solutions):
        """ Update mean of strategy. """
        x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
        y_k = (x_k - self.mean) / self.sigma  # ~ N(0, C)
        y_w = np.sum(y_k[: self.mu].T * self.weights[: self.mu], axis=1)
        mean = self.mean + self.c_m * self.sigma * y_w
        return y_k, y_w, mean

    def update_p_sigma(self, y_w):
        """ Update evolution path for covariance matrix. """
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None
        C_2 = B.dot(np.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
        p_sigma_new = (1 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * C_2.dot(y_w)
        return p_sigma_new, C_2

    def update_p_c(self, y_w):
        """ Update evolution path for sigma/stepsize. """
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        h_sigma_cond_left = norm_p_sigma / np.sqrt(
            1 - (1 - self.c_sigma) ** (2 * (self.generation + 1)))
        h_sigma_cond_right = (1.4 + 2 / (self.n_dim + 1)) * self.chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0
        p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(
             self.c_c * (2 - self.c_c) * self.mu_eff) * y_w
        return p_c, norm_p_sigma, h_sigma

    def update_covariance(self, y_k, h_sigma, C_2):
        """ Update covariance matrix estimator using a rank 1 + μ updates. """
        w_io = self.weights * np.where(self.weights >= 0, 1, self.n_dim/
                (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + self._EPS))
        delta_h_sigma = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        rank_one = np.outer(self.p_c, self.p_c)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0)
        C = ((1 + self.c_1 * delta_h_sigma
              - self.c_1
              - self.c_mu * np.sum(self.weights)) * self._C
             + self.c_1 * rank_one
             + self.c_mu * rank_mu)
        return C

    def update_sigma(self, norm_p_sigma):
        """ Update stepsize sigma. """
        sigma = self.sigma * np.exp((self.c_sigma / self.d_sigma)
                                    * (norm_p_sigma / self.chi_n - 1))
        return sigma


def check_cma_es_params(population_size, n_dim, sigma, c_1, c_mu, c_sigma, c_c):
    """ Check lrates and other params of CMA-ES. """
    assert population_size > 0, "popsize must be non-zero positive value."
    assert n_dim > 1, "The dimension of mean must be larger than 1"
    assert sigma > 0, "sigma must be non-zero positive value"
    assert c_1 <= 1 - c_mu, "invalid lrate for the rank-one update"
    assert c_mu <= 1 - c_1, "invalid lrate for the rank-μ update"
    assert c_sigma < 1, "invalid lrate for cum. of step-size c."
    assert c_c <= 1, "invalid lrate for cum. of rank-one update"
