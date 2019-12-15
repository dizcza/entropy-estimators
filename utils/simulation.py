import torch
import torch.utils.data
from utils.algebra import to_onehot


def log_softmax_data(n_samples=50000, n_classes=10, p_argmax=0.95, onehot=False):
    """
    Simulates readout activation (before softmax) of a typical neural network, trained to the accuracy of `p_argmax`.

    Parameters
    ----------
    n_samples : int
        No. of samples
    n_classes : int
        No. of unique classes
    p_argmax : float
        Network accuracy (softmax probability of the correctly predicted class).
    onehot : bool
        Return as one-hot encoded vectors (True) or a list of class indices (False).

    Returns
    -------
    x_data : torch.FloatTensor
        Simulated softmax probabilities of shape (n_samples, n_classes)
    y_labels : torch.LongTensor
        Ground truth labels (class indices) of shape
            * (n_samples,), if `onehot` is False
            * (n_samples, n_classes), otherwise
    """
    x_data = torch.randn(n_samples, n_classes)
    y_labels = x_data.argmax(dim=1)
    x_argmax = x_data[range(x_data.shape[0]), y_labels]
    softmax_sum = x_data.exp().sum(dim=1) - x_argmax
    x_argmax = torch.log(p_argmax * softmax_sum / (1 - p_argmax))
    x_data[range(x_data.shape[0]), y_labels] = x_argmax
    if onehot:
        y_labels = to_onehot(y_labels, n_classes=n_classes)
    return x_data, y_labels
