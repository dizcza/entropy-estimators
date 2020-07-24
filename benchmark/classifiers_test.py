from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import TensorDataset
from tqdm import tqdm, trange

from estimators import mine_mi, discrete_entropy, micd, \
    npeet_entropy, gcmi_entropy
from estimators import npeet_mi, gcmi_mi
from mighty.models import MLP
from mighty.utils.algebra import to_onehot
from utils.common import set_seed, timer_profile, Timer
from utils.constants import IMAGES_DIR


class Classifier:

    def __init__(self, hidden=64, verbose=False):
        self.hidden = hidden
        self.model = None
        self.batch_size = 254
        self.verbose = verbose

    def fit(self, x, y, n_epochs=20):
        x = torch.from_numpy(x).type(torch.float32)
        y = torch.LongTensor(y)
        n_features = x.shape[1]
        n_classes = y.unique().shape[0]
        self.model = MLP(n_features, self.hidden, n_classes)
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=1e-3, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if torch.cuda.is_available():
            self.model.cuda()

        for epoch in trange(n_epochs, disable=not self.verbose,
                            desc="Fitting the classifier"):
            for x_batch, y_batch in iter(loader):
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def predict_proba(self, x):
        assert self.model is not None
        self.model.eval()
        x = torch.from_numpy(x).type(torch.float32)
        dataset = TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                             shuffle=False)

        y_pred = []
        with torch.no_grad():
            for x_batch, in iter(loader):
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                outputs = self.model(x_batch)
                y_pred.append(outputs)

        y_pred = torch.cat(y_pred, dim=0)
        y_pred = y_pred.softmax(dim=1)
        return y_pred.numpy()


class GaussianMixture:
    def __init__(self, n_features=10, n_classes=5, spread=10, verbose=False):
        self.n_features = n_features
        self.n_classes = int(n_classes)
        self.spread = spread
        self.verbose = verbose
        self.loc = np.random.uniform(-self.spread, self.spread,
                                     size=(self.n_classes, self.n_features))
        scale_spread = np.sqrt(self.spread)
        self.scale = np.random.uniform(0.1 * scale_spread, scale_spread,
                                       size=(self.n_classes, self.n_features))

    def sample_gaussian_mixture(self, n_samples=1000):
        x_features = []
        y_labels = []
        while len(x_features) < n_samples:
            y = np.random.randint(0, self.n_classes, size=n_samples)
            loc_samples = self.loc[y]  # (n_samples, self.n_features)
            scale_samples = self.scale[y]
            x = np.random.normal(loc=loc_samples, scale=scale_samples,
                                 size=(n_samples, self.n_features))
            dist = np.linalg.norm(np.expand_dims(x, axis=1) - self.loc,
                                  axis=2)  # (n_samples, self.n_classes)
            assert dist.shape == (n_samples, self.n_classes)
            y_predicted = dist.argmin(axis=1)  # (n_samples,)
            correct_mask = y == y_predicted
            x = x[correct_mask]
            y = y[correct_mask]
            x_features.extend(x)
            y_labels.extend(y)
        x_features = np.vstack(x_features[: n_samples]).astype(np.float32)
        y_labels = np.array(y_labels[: n_samples])
        # given x, y is known (deterministic)
        mutual_info_true = discrete_entropy(y_labels)
        return x_features, y_labels, mutual_info_true

    def sample_softmax(self, n_samples=1000):
        def filter_correct(x_features, y_labels):
            y_proba = classifier.predict_proba(x_features)
            correct = y_proba.argmax(axis=1) == y_labels
            x_features = x_features[correct]
            return x_features

        x_features, y_labels, _ = self.sample_gaussian_mixture(n_samples)
        classifier = Classifier(verbose=self.verbose)
        classifier.fit(x_features, y_labels)
        x_features = filter_correct(x_features, y_labels)
        x_features = list(x_features)
        while len(x_features) < n_samples:
            x, y, _ = self.sample_gaussian_mixture(n_samples)
            x = filter_correct(x, y)
            x_features.extend(x)
        x_features = np.vstack(x_features)[: n_samples].astype(np.float32)
        y_proba = classifier.predict_proba(x_features)
        mutual_info_true = discrete_entropy(y_proba.argmax(axis=1))

        assert x_features.shape == (n_samples, self.n_features)
        assert y_proba.shape == (n_samples, self.n_classes)

        return x_features, y_proba, mutual_info_true


def plot_gmm(x=None, y=None):
    if x is None:
        gaussian_mixture = GaussianMixture(n_features=2, n_classes=3)
        x, y, _ = gaussian_mixture.sample_gaussian_mixture(n_samples=1000)
    else:
        assert x.ndim == 2, "Only 2D plots"
    if y.ndim == 2:
        y = y.argmax(axis=1)
    locations = []
    for class_id in np.unique(y):
        x_cluster = x[y == class_id]
        if len(x_cluster) == 0:
            print(class_id)
        loc = x_cluster.mean(axis=0)
        locations.append(loc)
        plt.scatter(x_cluster[:, 0], x_cluster[:, 1], alpha=0.25)
    locations = np.vstack(locations)
    n_classes = np.unique(y).shape[0]
    plt.scatter(locations[:, 0], locations[:, 1], marker='x')
    plt.title(f'Gaussian Mixture Model simulated data for {n_classes} classes')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()


class MutualInfoTest:

    def __init__(self, x, y, mi_true, verbose=True):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        self.x = x
        self.y = y
        self.mi_true = mi_true
        self.verbose = verbose
        if self.verbose:
            plot_gmm(x=self.x, y=self.y)

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
        if self.is_integer(self.y):
            return micd(x, self.y,
                        entropy_estimator=partial(npeet_entropy, k=k))
        y = self.normalize(self.y)
        return npeet_mi(x, y=y, k=k)

    @staticmethod
    def normalize(x):
        return (x - x.mean()) / x.std()

    @staticmethod
    def is_integer(y):
        return y.squeeze().ndim == 1 and np.issubdtype(y.dtype, np.integer)

    @staticmethod
    def add_noise(x):
        return x + np.random.normal(
            loc=0, scale=1e-5, size=x.shape).astype(np.float32)

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
        if self.is_integer(self.y):
            return micd(x, self.y, entropy_estimator=gcmi_entropy)
        y = self.normalize(self.y)
        y = self.add_noise(y)
        return gcmi_mi(x.T, y.T)

    @timer_profile
    def mine(self):
        x = self.normalize(self.x)
        y = self.y
        if self.y.ndim == 1:
            y = to_onehot(y)
        noise_std = 0.1 * np.std(x)
        return mine_mi(x, y, hidden_units=128, epochs=50, noise_std=noise_std,
                       verbose=self.verbose)

    def run_all(self):
        estimated = {}
        for estimator in (self.npeet, self.gcmi, self.mine):
            try:
                estimated[estimator.__name__] = estimator()
            except Exception as e:
                estimated[estimator.__name__] = np.nan
        if self.verbose:
            for estimator_name, estimator_value in estimated.items():
                print(
                    f"{estimator_name} I(X;Y)={estimator_value:.3f} (true value: {self.mi_true:.3f})")
        return estimated


def mi_test(method='sample_gaussian_mixture', n_samples=10000, n_features=10,
            parameters=np.linspace(2, 50, num=10), xlabel=''):
    estimated = defaultdict(list)
    for param in tqdm(parameters, desc=f"{method} test"):
        gaussian_mixture = GaussianMixture(n_features=n_features, n_classes=param)
        generator = getattr(gaussian_mixture, method)
        x, y, mi_true = generator(n_samples=n_samples)
        estimated_test = MutualInfoTest(x=x, y=y, mi_true=mi_true,
                                        verbose=False).run_all()
        estimated['true'].append(mi_true)
        for estimator_name, estimator_value in estimated_test.items():
            estimated[estimator_name].append(estimator_value)
    value_true = estimated.pop('true')
    plt.figure()
    plt.plot(parameters, value_true, label='true', ls='--', lw=2, marker='x')
    for estimator_name, estimator_value in estimated.items():
        plt.plot(parameters, estimator_value, label=estimator_name)
    plt.xlabel(xlabel)
    plt.ylabel('Estimated Mutual Information, bits')
    plt.title(
        f"{method}: len(X)={n_samples}, dim(X)={n_features}")
    plt.legend()
    # plt.savefig(IMAGES_DIR / f"{method}.png")
    plt.show()


def mi_test_verbose(n_samples=10000, n_features=2,
                    parameters=np.linspace(30, 50, num=10, dtype=int)):
    for param in tqdm(parameters, desc=f"test"):
        gaussian_mixture = GaussianMixture(n_features=n_features, n_classes=param, verbose=True)
        x, y, mi_true = gaussian_mixture.sample_softmax(n_samples)
        estimated_test = MutualInfoTest(x=x, y=y, mi_true=mi_true,
                                        verbose=True).gcmi()
        print(estimated_test)


if __name__ == '__main__':
    set_seed(26)
    # plot_gmm()
    mi_test(method='sample_softmax')
    Timer.checkpoint()
