from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from benchmark.entropy_test import generate_normal_correlated
from benchmark.mutual_information.mutual_information import MutualInfoTest
from utils.algebra import entropy_normal_theoretic
from utils.common import set_seed, Timer
from utils.constants import RESULTS_DIR

IMAGES_MUTUAL_INFO_DISTRIBUTIONS_DIR = RESULTS_DIR.joinpath(
    "mutual_information", "images", "distributions")


def _mi_squared_integers(n_samples, n_features, param):
    assert param >= 1
    x = np.random.randint(low=0, high=param + 1, size=(n_samples, n_features))
    y = x ** 2
    value_true = n_features * np.log2(param)
    return x, y, value_true


def _mi_uniform_squared(n_samples, n_features, param):
    assert param >= 1
    x = np.random.uniform(low=0, high=param, size=(n_samples, n_features))
    y = x ** 2
    # I(X; Y) = H(X) - H(X|Y) = H(X), since x=sqrt(y) is a deterministic func
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


def mi_test(generator, n_samples=1000, n_features=10,
            parameters=np.linspace(1, 50, num=10), xlabel=''):
    estimated = defaultdict(list)
    for param in tqdm(parameters, desc=f"{generator.__name__} test"):
        x, y, mi_true = generator(n_samples, n_features, param)
        estimated_test = MutualInfoTest(x=x, y=y, mi_true=mi_true).run_all()
        estimated['true'].append(mi_true)
        for estimator_name, estimator_value in estimated_test.items():
            estimated[estimator_name].append(estimator_value)
    value_true = estimated.pop('true')
    plt.figure()
    plt.plot(parameters, value_true, label='true', ls='--', marker='x', markersize=8)
    for estimator_name, estimator_value in estimated.items():
        plt.plot(parameters, estimator_value, label=estimator_name)
    plt.xlabel(xlabel)
    plt.ylabel('Estimated Mutual Information, bits')
    plt.title(f"{generator.__name__.lstrip('_mi_')}: len(X)={n_samples}, dim(X)={n_features}")
    plt.legend()
    plt.savefig(IMAGES_MUTUAL_INFO_DISTRIBUTIONS_DIR / f"{generator.__name__}.png")
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
    Timer.checkpoint()


if __name__ == '__main__':
    set_seed(119)
    mi_all_tests()
