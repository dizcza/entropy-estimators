from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

from estimators import mine_mi, npeet_entropy, gcmi_entropy, micd
from utils.algebra import to_onehot, mutual_info_upperbound
from utils.common import set_seed
from utils.simulation import log_softmax_data


class SoftmaxTest:

    def __init__(self, n_samples=20000, n_classes=10, p_argmax=0.99, verbose=True):
        self.n_classes = n_classes
        self.verbose = verbose
        self.softmax, self.labels = log_softmax_data(n_samples=n_samples, n_classes=n_classes, p_argmax=p_argmax)

        self.mutual_info_true = mutual_info_upperbound(accuracy=p_argmax, n_classes=n_classes)

    def mine(self, hidden_units=64, noise_variance=0., epochs=30):
        """
        Mutual Information Neural Estimation (MINE) lower-bound.

        Parameters
        ----------
        hidden_units : int
            Dimensionality of the hidden layer.
        noise_variance : float
            Noise variance.
        epochs : int
            No. of training epochs.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        labels = to_onehot(self.labels)
        estimated = mine_mi(self.softmax, y=labels, hidden_units=hidden_units,
                            noise_variance=noise_variance, epochs=epochs, verbose=self.verbose)
        return estimated

    def kmeans(self, n_clusters=None):
        """
        The simplest binning estimator, based on KMeans clustering.

        Parameters
        ----------
        n_clusters : int
            No. of KMeans clusters.
            For the reliable estimate, should be greater or equal to `self.n_classes`.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        if n_clusters is None:
            n_clusters = self.n_classes
        model = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=256)
        softmax_binned = model.fit_predict(self.softmax)
        estimated = mutual_info_score(softmax_binned, self.labels)
        # convert nats to bits
        estimated *= np.log2(np.e)
        return estimated

    def npeet(self):
        """
        Non-parametric Entropy Estimation Toolbox.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        return micd(self.softmax.numpy(), self.labels.numpy(), entropy_estimator=npeet_entropy)

    def gcmi(self):
        """
        Gaussian-Copula Mutual Information.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        return micd(self.softmax.numpy(), self.labels.numpy(), entropy_estimator=gcmi_entropy)

    def run_all(self):
        mi_estimated = {}
        for estimator in (self.mine, self.kmeans, self.npeet):
            mi_estimated[estimator.__name__] = estimator()
        if self.verbose:
            for estimator_name, estimator_value in mi_estimated.items():
                print(f"{estimator_name} I(X;Y)={estimator_value:.3f} (true value: {self.mutual_info_true:.3f})")
        return mi_estimated


def softmax_test(n_classes=10):
    estimated = defaultdict(list)
    model_accuracies = np.linspace(0.9, 1.0, num=3, endpoint=True)
    for model_accuracy in tqdm(model_accuracies, desc="Testing softmax"):
        estimated_test = SoftmaxTest(n_classes=n_classes, p_argmax=model_accuracy, verbose=False).run_all()
        for estimator_name, estimator_value in estimated_test.items():
            estimated[estimator_name].append(estimator_value)
    for estimator_name, estimator_value in estimated.items():
        plt.plot(model_accuracies, estimator_value, label=estimator_name)
    mi_true = mutual_info_upperbound(model_accuracies, n_classes=n_classes)
    plt.plot(model_accuracies, mi_true, label='upperbound', ls='--')
    plt.xlim(xmax=1.0)
    plt.xlabel('Softmax accuracy')
    plt.ylabel('I(X;Y)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    set_seed(26)
    # SoftmaxTest().mine()
    softmax_test()
