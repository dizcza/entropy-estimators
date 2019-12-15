import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data

from utils.algebra import exponential_moving_average


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
        self.mutual_info_history.append(mutual_info_lower_bound.item())
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
        # convert nats to bits
        self.mutual_info_history = np.multiply(self.mutual_info_history, np.log2(np.e))

    def get_mutual_info(self):
        """
        Returns
        -------
        float
            Estimated lower bound of mutual information as the mean of the last quarter history points.
        """
        fourth_quantile = self.mutual_info_history[-len(self.mutual_info_history) // 4:]
        return np.mean(fourth_quantile)
