import warnings

import numpy as np
from functools import partial

from estimators import mine_mi, npeet_mi, gcmi_mi, micd, npeet_entropy
from utils.common import timer_profile


class MutualInfoTest:

    def __init__(self, x, y, mi_true, verbose=False):
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
            See https://github.com/gregversteeg/NPEET/blob/master/npeet_doc.pdf

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        x = self.normalize(self.x)
        if self.is_univariate_integer(self.y):
            return micd(x, self.y,
                        entropy_estimator=partial(npeet_entropy, k=k))
        y = self.normalize(self.y)
        return npeet_mi(x, y=y, k=k)

    @staticmethod
    def normalize(x):
        return (x - x.mean()) / x.std()

    @staticmethod
    def is_univariate_integer(y):
        # check if 'y' is an array of ints
        return y.squeeze().ndim == 1 and np.issubdtype(y.dtype, np.integer)

    @staticmethod
    def add_noise(x):
        return x + np.random.normal(
            loc=0, scale=1e-7, size=x.shape).astype(np.float32)

    @staticmethod
    def nats2bits(entropy):
        return entropy * np.log2(np.e)

    def _gcmi(self, backend=gcmi_mi):
        from estimators.gcmi.python.gcmi import gcmi_model_cd, gcmi_mixture_cd

        x = self.x
        if np.issubdtype(x.dtype, np.integer):
            x = self.add_noise(x)

        if self.is_univariate_integer(self.y):
            # micd(x, self.y, entropy_estimator=gcmi_entropy) is not stable:
            # The Cholesky decomposition sometimes cannot be computed from
            # a subset of X data: x|y=yi. In this case, an error "Matrix is
            # not positive definite" is thrown.
            if backend is gcmi_mi:
                backend_cd = gcmi_model_cd
            else:
                backend_cd = gcmi_mixture_cd
            return backend_cd(x.T, self.y, Ym=len(np.unique(self.y)))

        y = self.y
        if np.issubdtype(y.dtype, np.integer):
            y = self.add_noise(y)
        return backend(x.T, y.T)

    @timer_profile
    def gcmi_gaussian_copula(self):
        """
        Gaussian-Copula Mutual Information.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        return self._gcmi(backend=gcmi_mi)

    @timer_profile
    def gcmi_gaussian(self):
        """
        Mutual information between two Gaussian variables.

        Returns
        -------
        float
            Estimated I(X;Y).

        """
        from estimators.gcmi.python.gcmi import mi_gg
        return self._gcmi(backend=mi_gg)

    @timer_profile
    def mine(self):
        x = self.normalize(self.x)
        y = self.normalize(self.y)
        return mine_mi(x, y, hidden_units=128, epochs=30,
                       noise_std=1e-4, verbose=self.verbose)

    @timer_profile
    def idtxl_kraskov(self):
        from estimators.IDTxl.idtxl.estimators_jidt import JidtKraskovMI
        from estimators.IDTxl.idtxl.estimators_opencl import OpenCLKraskovMI
        settings = {'kraskov_k': 4}
        try:
            estimator = OpenCLKraskovMI(settings=settings)
        except RuntimeError:
            warnings.warn("No OpenCL backed detected. Run "
                          "'conda install -c conda-forge pyopencl' "
                          "in a terminal.")
            estimator = JidtKraskovMI(settings=settings)

        # IDTxl requires 64-bit float precision
        x = self.x.astype(np.float64)
        y = self.y.astype(np.float64)

        mi_nats = estimator.estimate(x, y)
        if isinstance(mi_nats, np.ndarray):
            # fix for OpenCLKraskovMI
            assert mi_nats.shape == (1,)
            mi_nats = mi_nats.item()
        mi_bits = self.nats2bits(mi_nats)
        return mi_bits

    @timer_profile
    def idtxl_gaussian(self):
        from estimators.IDTxl.idtxl.estimators_jidt import JidtGaussianMI
        estimator = JidtGaussianMI()

        # IDTxl requires 64-bit float precision
        x = self.x.astype(np.float64)
        y = self.y.astype(np.float64)

        mi_nats = estimator.estimate(x, y)
        mi_bits = self.nats2bits(mi_nats)
        return mi_bits

    def run_all(self):
        estimated = {}
        for estimator in (self.npeet, self.gcmi_gaussian_copula,
                          self.gcmi_gaussian, self.mine, self.idtxl_kraskov,
                          self.idtxl_gaussian):
            try:
                estimated[estimator.__name__] = estimator()
            except Exception as e:
                estimated[estimator.__name__] = np.nan
        if self.verbose:
            for estimator_name, estimator_value in estimated.items():
                print(f"{estimator_name} I(X;Y)={estimator_value:.3f} "
                      f"(true value: {self.mi_true:.3f})")
        return estimated
