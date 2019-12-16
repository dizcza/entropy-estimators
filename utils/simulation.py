import torch
import torch.distributions
import torch.utils.data
from utils.algebra import to_onehot


def calc_accuracy(y_true, y_pred):
    return (y_true == y_pred).type(torch.float32).mean()


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
    y_labels : torch.FloatTensor
        Ground truth labels (class indices) of shape
            * (n_samples,), if `onehot` is False
            * (n_samples, n_classes), otherwise
    """
    p_err = (1. - p_argmax) / (n_classes - 1)
    proba_noise = torch.distributions.uniform.Uniform(low=0.5 * p_err, high=1.5 * p_err + 1e-6)
    n_samples_correct = int(p_argmax * n_samples)
    n_samples_incorrect = n_samples - n_samples_correct
    x_correct = proba_noise.sample((n_samples_correct, n_classes))
    y_correct = torch.randint(low=0, high=n_classes, size=(n_samples_correct,))
    x_correct[range(x_correct.shape[0]), y_correct] += (p_argmax - p_err)

    x_incorrect = torch.rand(n_samples_incorrect, n_classes)
    x_sorted, argsort = x_incorrect.sort(dim=1, descending=True)
    y_incorrect = argsort[:, 1]

    x = torch.cat([x_correct, x_incorrect], dim=0)
    y = torch.cat([y_correct, y_incorrect], dim=0)

    x.log_()
    accuracy_true = calc_accuracy(y_true=x.argmax(dim=1), y_pred=y)
    print(f"Softmax accuracy generated={accuracy_true:.3f}, goal={p_argmax:.3f}")

    if onehot:
        y = to_onehot(y, n_classes=n_classes)

    return x, y
