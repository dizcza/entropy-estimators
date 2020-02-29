import numpy as np
import torch


def exponential_moving_average(array, window):
    """
    Exponential moving average in a sliding window.

    Parameters
    ----------
    array : (N,) np.ndarray
        Input array-like.
    window : int
        Sliding window width.

    Returns
    -------
    out : (N,) np.ndarrat
        Filtered array of the same length.
    """
    array = np.asarray(array)
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = array.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = array[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = array * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def to_onehot(y_labels, n_classes=None):
    if n_classes is None:
        n_classes = len(y_labels.unique(sorted=False))
    y_onehot = torch.zeros(y_labels.shape[0], n_classes, dtype=torch.int64)
    y_onehot[torch.arange(y_onehot.shape[0]), y_labels] = 1
    return y_onehot


def mutual_info_upperbound(accuracy, n_classes):
    """
    Estimates the mutual information upper bound as :math:`log_2 n_classes` with the correction for the training
    degree `accuracy`.
    See https://colab.research.google.com/drive/124gIEjgF0HXOObG33R4rbpCyb5CLQ8UT#scrollTo=UbHXS3rB4IAt

    Parameters
    ----------
    accuracy : np.ndarray or float
        The model accuracy.
    n_classes : int
        No. of classes in a dataset.

    Returns
    -------
    np.ndarray or float
        Mutual information upper bound for a model, trained to the accuracy of `accuracy` on a dataset with
        `n_classes` unique classes.
    """
    entropy_correct = accuracy * np.log2(1. / accuracy)
    entropy_incorrect = (1. - accuracy) * np.log2((n_classes - 1) / (1. - accuracy + 1e-10))
    noise_entropy = entropy_correct + entropy_incorrect
    return np.log2(n_classes) - noise_entropy


def entropy_normal_theoretic(cov):
    assert cov.shape[0] == cov.shape[1]
    n_features = cov.shape[0]
    logdet = np.linalg.slogdet(cov)[1] * np.log2(np.e)
    value_true = 0.5 * (n_features * np.log2(2 * np.pi * np.e) + logdet)
    return value_true
