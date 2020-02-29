"""
Mutual Information Neural Estimation (MINE).
"""
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from tqdm import trange

from utils.algebra import exponential_moving_average
from utils.constants import BATCH_SIZE


class MutualInfoNeuralEstimationNetwork(nn.Module):
    """
    https://arxiv.org/pdf/1801.04062.pdf
    """

    def __init__(self, x_size: int, y_size: int, hidden_units=64):
        """
        A network to estimate I(X; Y).

        Parameters
        ----------
        x_size, y_size : int
            Sizes of X and Y.
        hidden_units : int
            Hidden layer size.
        """
        super().__init__()
        self.fc_x = nn.Linear(x_size, hidden_units, bias=False)
        self.fc_y = nn.Linear(y_size, hidden_units, bias=False)
        self.xy_bias = nn.Parameter(torch.zeros(hidden_units), requires_grad=True)
        self.fc_output = nn.Linear(hidden_units, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x, y : torch.Tensor
            Data batches.

        Returns
        -------
        output : torch.Tensor
            Kullback-Leibler member for I(X; Y) estimation.
        """
        hidden = self.relu(self.fc_x(x) + self.fc_y(y) + self.xy_bias)
        output = self.fc_output(hidden)
        return output


class MutualInfoNeuralEstimationTrainer:

    log2_e = np.log2(np.e)

    def __init__(self, mine_model, learning_rate=1e-3):
        """
        Parameters
        ----------
        mine_model : MutualInfoNeuralEstimationNetwork
            A network to estimate mutual information.
        learning_rate : float
            Optimizer learning rate.
        """
        if torch.cuda.is_available():
            mine_model = mine_model.cuda()
        self.mine_model = mine_model
        self.optimizer = torch.optim.Adam(self.mine_model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.mutual_info_history = [0]

    def __repr__(self):
        return f"{MutualInfoNeuralEstimationTrainer.__name__}(mine_model={self.mine_model}, optimizer={self.optimizer})"

    def start_training(self):
        self.mutual_info_history = [0]

    def train_batch(self, x_batch, y_batch):
        """
        Performs a single step to refine I(X; Y).

        Parameters
        ----------
        x_batch, y_batch : torch.Tensor
            A batch of multidimensional X and Y of size (B, N) to estimate mutual information from.
            N could be 1 or more.
        """
        if torch.cuda.is_available():
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        self.optimizer.zero_grad()
        pred_joint = self.mine_model(x_batch, y_batch)
        y_batch = y_batch[torch.randperm(y_batch.shape[0], device=y_batch.device)]
        pred_marginal = self.mine_model(x_batch, y_batch)
        mutual_info_lower_bound = pred_joint.mean() - pred_marginal.exp().mean().log()
        mi_bits = mutual_info_lower_bound.item() * self.log2_e  # convert nats to bits
        self.mutual_info_history.append(mi_bits)
        loss = -mutual_info_lower_bound  # maximize
        loss.backward()
        self.optimizer.step()

    def finish_training(self, filter_size=30, filter_rounds=3):
        """
        Parameters
        ----------
        filter_size : int
            Smoothing filter window size.
        filter_rounds : int
            How many times to apply.
        """
        for repeat in range(filter_rounds):
            self.mutual_info_history = exponential_moving_average(self.mutual_info_history,
                                                                  window=filter_size)

    def _smooth(self, filter_size=30):
        return exponential_moving_average(self.mutual_info_history, window=filter_size)

    def show_history(self):
        plt.plot(self.mutual_info_history, alpha=0.2, label='raw')
        mi_smooth = self._smooth()
        mi_argmax = mi_smooth.argmax()
        plt.plot(mi_smooth, linestyle='--', color='#ff7f0e', label='filtered')
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("I(X;Y), bits")
        plt.title(f"max(I) = {mi_smooth[mi_argmax]:.2f} bits")
        plt.scatter(mi_argmax, mi_smooth[mi_argmax], c='#ff7f0e', marker='D')
        plt.show()

    def get_mutual_info(self):
        """
        Returns
        -------
        float
            Estimated mutual information lower bound.
        """
        return self._smooth().max()


def mine_mi(x, y, hidden_units=64, noise_variance=0, epochs=10, tol=1e-2, verbose=False):
    """
    MINE estimation of I(X;Y).

    Parameters
    ----------
    x, y : torch.Tensor
        Realizations of X and Y multidimensional random variables of sizes (N, xdim) and (N, ydim).
    hidden_units : int
        Dimensionality of the hidden layer.
    noise_variance : float
        Noise variance.
    epochs : int
        No. of training epochs on the same data.
    tol : float
        Tolerance of the estimator. If 5 successive epochs don't improve the best estimate by more than `tol`,
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
    normal_sampler = torch.distributions.normal.Normal(loc=0, scale=np.sqrt(noise_variance))
    mine_net = MutualInfoNeuralEstimationNetwork(x_size=x.shape[1], y_size=y.shape[1], hidden_units=hidden_units)
    mine_trainer = MutualInfoNeuralEstimationTrainer(mine_net)

    mi_last = deque(maxlen=5)
    mi_last.append(0.)
    mine_trainer.start_training()
    for epoch in trange(epochs, desc='Optimizing MINE', disable=not verbose):
        permutation = torch.randperm(x.shape[0])
        x_perm = x[permutation].split(BATCH_SIZE)
        y_perm = y[permutation].split(BATCH_SIZE)
        for x_batch, y_batch in zip(x_perm, y_perm):
            y_batch = y_batch + normal_sampler.sample(y_batch.shape)
            mine_trainer.train_batch(x_batch=x_batch, y_batch=y_batch)
        mi_curr = mine_trainer.get_mutual_info()
        mi_last.append(mi_curr)
        if np.std(mi_last) < tol:
            break

    if verbose:
        mine_trainer.show_history()

    return mine_trainer.get_mutual_info()
