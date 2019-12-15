import numpy as np
from sklearn import cluster
from sklearn.metrics import mutual_info_score

from estimators import mine_mi, npeet_entropy, gcmi_entropy, micd
from utils.algebra import to_onehot, mutual_info_upperbound
from utils.common import set_seed
from utils.simulation import log_softmax_data


class SoftmaxTest:

    def __init__(self, n_samples=20000, n_classes=10, p_argmax=0.99):
        self.n_classes = n_classes
        self.softmax, self.labels = log_softmax_data(n_samples=n_samples, n_classes=n_classes, p_argmax=p_argmax)
        self.mutual_info_true = mutual_info_upperbound(accuracy=p_argmax, n_classes=n_classes)

    def mine(self, hidden_units=64, noise_variance=0.):
        """
        Mutual Information Neural Estimation (MINE) lower-bound.

        Parameters
        ----------
        hidden_units : int
            Dimensionality of the hidden layer.
        noise_variance : float
            Noise variance.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        labels = to_onehot(self.labels)
        estimated = mine_mi(self.softmax, y=labels, hidden_units=hidden_units,
                            noise_variance=noise_variance, show=True)
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
        model = cluster.KMeans(n_clusters=n_clusters, n_jobs=-1)
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
        for estimator in (self.mine, self.kmeans, self.npeet, self.gcmi):
            mi_estimated = estimator()
            print(f"{estimator.__name__} I(X;Y)={mi_estimated:.3f} (true value: {self.mutual_info_true:.3f})")


if __name__ == '__main__':
    set_seed(26)
    test = SoftmaxTest()
    test.run_all()
