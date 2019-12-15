import numpy as np
from estimators import npeet_entropy, gcmi_entropy


def micd(x, y, entropy_estimator=npeet_entropy):
    """
    Mutual information estimator between multidimensional continuous X and 1-d discrete Y.

    Parameters
    ----------
    x : np.ndarray
        (N, xdim) xdim-dimensional array of N realizations of continuous random variable X.
    y : np.ndarray
        (N,) 1-d list of N realizations of discrete random variable Y.
    entropy_estimator : callable
        Entropy estimator function.

    Returns
    -------
    float
        Estimated I(X; Y) in bits.
    """
    axis = 0
    if entropy_estimator is gcmi_entropy:
        x = x.T
        axis = 1
    entropy_x = entropy_estimator(x)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        mask = y == yval
        x_given_y = np.take(x, mask.nonzero()[0], axis=axis)
        entropy_x_given_y += py * entropy_estimator(x_given_y)
    return entropy_x - entropy_x_given_y
