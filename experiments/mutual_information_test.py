from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from estimators import mine_mi, npeet_mi, gcmi_mi
from experiments.entropy_test import generate_normal_correlated
from utils.algebra import entropy_normal_theoretic
from utils.common import set_seed, timer_profile, Timer
from utils.constants import IMAGES_DIR, TIMINGS_DIR


class MITest:

    def __init__(self, x, y, mi_true, verbose=True):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        self.x = x
        self.y = y
        self.mi_true = mi_true
        self.verbose = verbose

    @timer_profile
    def npeet(self, k=3):
        """
        Non-parametric Entropy Estimation Toolbox.

        Parameters
        ----------
        k : int
            No. of nearest neighbors.
            See https://github.com/gregversteeg/NPEET/blob/master/npeet_doc.pdf.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        x = self.normalize(self.x)
        y = self.normalize(self.y)
        return npeet_mi(x, y, k=k)

    @staticmethod
    def normalize(x):
        return (x - x.mean()) / x.std()

    @staticmethod
    def add_noise(x):
        return x + np.random.normal(loc=0, scale=1e-9, size=x.shape)

    @timer_profile
    def gcmi(self):
        """
        Gaussian-Copula Mutual Information.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        x = self.normalize(self.x)
        y = self.normalize(self.y)
        x = self.add_noise(x.T)
        y = self.add_noise(y.T)
        return gcmi_mi(x, y)

    @timer_profile
    def mine(self):
        x = self.normalize(self.x)
        y = self.normalize(self.y)
        noise_std = 0.1 * np.std(x)
        return mine_mi(x, y, hidden_units=128, epochs=50, noise_std=noise_std, verbose=self.verbose)

    def run_all(self):
        estimated = {}
        for estimator in (self.npeet, self.gcmi, self.mine):
            try:
                estimated[estimator.__name__] = estimator()
            except Exception as e:
                estimated[estimator.__name__] = np.nan
        if self.verbose:
            for estimator_name, estimator_value in estimated.items():
                print(f"{estimator_name} I(X;Y)={estimator_value:.3f} (true value: {self.mi_true:.3f})")
        return estimated


def _mi_squared_integers(n_samples, n_features, param):
    assert param >= 1
    x = np.random.randint(low=0, high=param + 1, size=(n_samples, n_features))
    y = x ** 2
    value_true = n_features * np.log2(param)
    return x, y, value_true


def _mi_uniform_squared(n_samples, n_features, param):
    assert param >= 1
    x = np.random.uniform(low=0, high=param, size=(n_samples, n_features))
    y = x ** 2  # y ~ U[0, param ^ 2]
    value_true = n_features * np.log2(param)
    return x, y, value_true


def _mi_normal_correlated(n_samples, n_features, param, loc=None):
    # 2*n_features to generate both X and Y
    xy, h_xy, cov_xy = generate_normal_correlated(n_samples, 2 * n_features, sigma=param, loc=loc)
    x, y = np.split(xy, 2, axis=1)
    cov_x = cov_xy[:n_features, :n_features]
    cov_y = cov_xy[n_features:, n_features:]
    h_x = entropy_normal_theoretic(cov_x)
    h_y = entropy_normal_theoretic(cov_y)
    value_true = h_x + h_y - h_xy
    return x, y, value_true


def _mi_normal_different_location(n_samples, n_features, param):
    # generate Multivariate Normal samples with sigma=10 and uniform locations
    loc = np.random.uniform(low=0, high=param, size=2 * n_features)
    return _mi_normal_correlated(n_samples=n_samples, n_features=n_features, param=10, loc=loc)


def _mi_additive_normal_noise(n_samples, n_features, param):
    x, h_x, cov_x = generate_normal_correlated(n_samples, n_features, param)
    noise, h_noise, cov_noise = generate_normal_correlated(n_samples, n_features, 0.5 * param)
    y = x + noise
    h_y = entropy_normal_theoretic(cov_x + cov_noise)
    value_true = h_y - h_noise
    return x, y, value_true


def mi_test(generator, n_samples=1000, n_features=10, parameters=np.linspace(1, 50, num=10), xlabel=''):
    estimated = defaultdict(list)
    for param in tqdm(parameters, desc=f"{generator.__name__} test"):
        x, y, mi_true = generator(n_samples, n_features, param)
        estimated_test = MITest(x=x, y=y, mi_true=mi_true, verbose=False).run_all()
        estimated['true'].append(mi_true)
        for estimator_name, estimator_value in estimated_test.items():
            estimated[estimator_name].append(estimator_value)
    value_true = estimated.pop('true')
    plt.figure()
    plt.plot(parameters, value_true, label='true', ls='--', marker='x')
    for estimator_name, estimator_value in estimated.items():
        plt.plot(parameters, estimator_value, label=estimator_name)
    plt.xlabel(xlabel)
    plt.ylabel('Estimated Mutual Information, bits')
    plt.title(f"{generator.__name__.lstrip('_mi_')}: len(X)={n_samples}, dim(X)={n_features}")
    plt.legend()
    plt.savefig(IMAGES_DIR / f"{generator.__name__}.png")
    # plt.show()


def mi_all_tests(n_samples=10_000, n_features=10):
    set_seed(26)
    mi_test(_mi_uniform_squared, n_samples=n_samples, n_features=n_features,
            xlabel=r'$X \sim $Uniform$(0, x); Y = X^2$')
    mi_test(_mi_squared_integers, n_samples=n_samples, n_features=n_features,
            xlabel=r'$X \sim $Randint$(0, x); Y = X^2$')
    mi_test(_mi_normal_correlated, n_samples=n_samples, n_features=n_features,
            xlabel=r'$XY \sim \mathcal{N}(0, \Sigma^\top \Sigma), \Sigma_{ij} \sim $Uniform$(0, x)$')
    mi_test(_mi_additive_normal_noise, n_samples=n_samples, n_features=n_features,
            xlabel=r'$X \sim \mathcal{N}(0, x^2); Y = X + \epsilon,'
                   r'\epsilon \sim \mathcal{N}(0,\left(\frac{x}{2}\right)^2$)')
    mi_test(_mi_normal_different_location, n_samples=n_samples, n_features=n_features,
            xlabel=r'$XY \sim \mathcal{N}(\mu, 10^2), \mu \sim $Uniform$(0, x)$')
    Timer.checkpoint(fpath=TIMINGS_DIR / "mutual_information.txt")


if __name__ == '__main__':
    mi_all_tests()
