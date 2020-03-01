import math
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
from tqdm import tqdm

from estimators import mine_mi, npeet_mi, gcmi_mi, discrete_mi, discrete_entropy
from experiments.entropy_test import generate_normal_correlated
from utils.algebra import entropy_normal_theoretic
from utils.common import set_seed, timer_profile, Timer
from utils.constants import IMAGES_DIR


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
        return x + np.random.normal(loc=0, scale=1e-10, size=x.shape)

    @timer_profile
    def gcmi(self):
        """
        Gaussian-Copula Mutual Information.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        x = self.add_noise(self.x.T)
        y = self.add_noise(self.y.T)
        return gcmi_mi(x, y)

    @timer_profile
    def kmeans(self, n_clusters=None):
        x = self.normalize(self.x)
        y = self.normalize(self.y)
        xy = np.c_[x, y]

        def _kmeans(n_cl):
            cluster = sklearn.cluster.KMeans(n_clusters=n_cl, n_init=2, max_iter=10)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                x_clusters = cluster.fit_predict(x)
                y_clusters = cluster.fit_predict(y)
                xy_clusters = cluster.fit_predict(xy)
            h_x = discrete_entropy(x_clusters)
            h_y = discrete_entropy(y_clusters)
            h_xy = discrete_entropy(xy_clusters)
            return h_x + h_y - h_xy

        if n_clusters is None:
            n_clusters_max = math.ceil(math.sqrt(x.shape[0]))
            n_clusters_max = max(2, n_clusters_max)
            n_candidates = np.arange(n_clusters_max, 2*n_clusters_max + 1, step=1, dtype=int)
            n_candidates = np.arange(n_clusters_max, x.shape[0] // 2, step=50)
            mi = {n_cl: _kmeans(n_cl) for n_cl in n_candidates}
            if self.verbose:
                plt.plot(list(mi.keys()), list(mi.values()))
                plt.show()
            return max(mi.values())

        return _kmeans(n_clusters)

    @timer_profile
    def mine(self):
        x = self.normalize(self.x)
        y = self.normalize(self.y)
        return mine_mi(x, y, hidden_units=128, epochs=50, verbose=self.verbose)

    def run_all(self):
        estimated = {}
        for estimator in (self.npeet, self.gcmi, self.mine, self.kmeans):
            try:
                estimated[estimator.__name__] = estimator()
            except Exception as e:
                estimated[estimator.__name__] = np.nan
        if self.verbose:
            for estimator_name, estimator_value in estimated.items():
                print(f"{estimator_name} I(X;Y)={estimator_value:.3f} (true value: {self.mi_true:.3f})")
        return estimated


def _mi_squared_integers(n_samples, n_features, param):
    x = np.random.randint(low=0, high=param + 1, size=(n_samples, n_features))
    y = x ** 2
    value_true = n_features * np.log2(param)
    return x, y, value_true


def _mi_normal_correlated(n_samples, n_features, param):
    xy, h_xy, cov_xy = generate_normal_correlated(n_samples, 2 * n_features, param)
    x, y = np.split(xy, 2, axis=1)
    cov_x = cov_xy[:n_features, :n_features]
    cov_y = cov_xy[n_features:, n_features:]
    h_x = entropy_normal_theoretic(cov_x)
    h_y = entropy_normal_theoretic(cov_y)
    value_true = h_x + h_y - h_xy
    return x, y, value_true


def _mi_additive_noise(n_samples, n_features, param):
    x, h_x, cov_x = generate_normal_correlated(n_samples, n_features, param)
    noise, h_noise, cov_noise = generate_normal_correlated(n_samples, n_features, 0.5 * param)
    y = x + noise
    h_y = entropy_normal_theoretic(cov_x + cov_noise)
    value_true = h_y - h_noise
    return x, y, value_true


def mi_test(generator, n_samples=1000, n_features=10, parameters=np.linspace(1, 50, num=10), xlabel=''):
    estimated = defaultdict(list)
    for param in tqdm(parameters, desc="entropy_test"):
        x, y, mi_true = generator(n_samples, n_features, param)
        # MITest(x=x, y=y, mi_true=mi_true, verbose=True).mine()
        estimated_test = MITest(x=x, y=y, mi_true=mi_true, verbose=False).run_all()
        estimated['true'].append(mi_true)
        for estimator_name, estimator_value in estimated_test.items():
            estimated[estimator_name].append(estimator_value)
    value_true = estimated.pop('true')
    plt.figure()
    plt.plot(parameters, value_true, label='true', ls='--')
    for estimator_name, estimator_value in estimated.items():
        plt.plot(parameters, estimator_value, label=estimator_name)
    plt.xlabel(xlabel)
    plt.ylabel('Estimated Mutual Information, bits')
    plt.title(f"{generator.__name__}, size ({n_samples},{n_features})")
    plt.legend()
    plt.savefig(IMAGES_DIR / f"{generator.__name__}.png")
    plt.show()


if __name__ == '__main__':
    set_seed(26)
    # mi_test(_mi_squared_integers, n_samples=1000, n_features=5, xlabel='X ~ [0, x]; Y = X ^ 2')
    # mi_test(_mi_normal_correlated, n_samples=1000, n_features=5, xlabel='XY ~ N(0, cov); cov ~ x')
    mi_test(_mi_additive_noise, n_samples=1000, n_features=5, xlabel='X ~ N(0, x); Y = X + Noise')

    Timer.checkpoint()
