"""
Mutual Information Neural Estimation (MINE).
"""
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions
from tqdm import trange

from mighty.monitor.mutual_info.neural_estimation import MINE_Net, MINE_Trainer
from mighty.utils.constants import BATCH_SIZE


class MINE_TrainerMatplot(MINE_Trainer):

    def show_history(self):
        fig, ax = plt.subplots()
        ax.plot(self.mi_history, alpha=0.2, label='raw')
        mi_smooth = self._smooth()
        mi_argmax = mi_smooth.argmax()
        ax.plot(mi_smooth, linestyle='--', color='#ff7f0e', label='filtered')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("I(X;Y), bits")
        ax.set_title(f"max(I) = {mi_smooth[mi_argmax]:.2f} bits")
        ax.scatter(mi_argmax, mi_smooth[mi_argmax], c='#ff7f0e', s=80, marker='x')
        ax.text(0.5, 0.05, s=repr(self.mine_model), fontsize=5, transform=ax.transAxes
                 )
        plt.legend()
        plt.show()

    def get_mutual_info(self):
        """
        Returns
        -------
        float
            Estimated mutual information lower bound.
        """
        return self._smooth().max()


def mine_mi(x, y, hidden_units=64, noise_std=0., epochs=10,
            batch_size=BATCH_SIZE, tol=1e-2, verbose=False):
    """
    MINE estimation of I(X;Y).

    Parameters
    ----------
    x, y : torch.Tensor
        Realizations of X and Y multidimensional random variables of sizes
        (N, xdim) and (N, ydim).
    hidden_units : int or tuple of int
        Dimensionalities of the hidden layer.
    noise_std : float
        Additive noise standard deviation (scale).
    epochs : int
        No. of training epochs on the same data.
    batch_size : int
        The batch size.
    tol : float
        Tolerance of the estimator. If 5 successive epochs don't improve the
        best estimate by more than `tol`,
        the training stops.
    verbose : bool
        Show the training progress and the history plot or not.

    Returns
    -------
    float:
        Estimated I(X;Y).

    """
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    normal_sampler = torch.distributions.normal.Normal(loc=0, scale=noise_std)
    mine_net = MINE_Net(x_size=x.shape[1], y_size=y.shape[1],
                        hidden_units=hidden_units)
    mine_trainer = MINE_TrainerMatplot(mine_net)

    mi_last = deque(maxlen=5)
    mi_last.append(0.)
    mine_trainer.reset()
    for epoch in trange(epochs, desc='Optimizing MINE', disable=not verbose):
        permutation = torch.randperm(x.shape[0])
        x_perm = x[permutation].split(batch_size)
        y_perm = y[permutation].split(batch_size)
        for x_batch, y_batch in zip(x_perm, y_perm):
            y_batch = y_batch + normal_sampler.sample(y_batch.shape)
            mine_trainer.train_batch(x_batch=x_batch, y_batch=y_batch)
        mi_curr = mine_trainer.get_mutual_info()
        mi_last.append(mi_curr)
        if len(mi_last) == mi_last.maxlen and np.std(mi_last) < tol:
            break

    if verbose:
        mine_trainer.show_history()

    return mine_trainer.get_mutual_info()
