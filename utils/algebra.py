import numpy as np


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


def nearestPD(matrix):
    """
    Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    if isPD(matrix):
        return matrix

    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(matrix))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(matrix.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


if __name__ == '__main__':
    for i in range(2, 10):
        for j in range(2, 100):
            A = np.random.randn(j, i)
            B = nearestPD(A)
            assert (isPD(B))
    print('unit test passed!')
