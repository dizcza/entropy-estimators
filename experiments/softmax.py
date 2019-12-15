import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions
import torch.utils.data
from sklearn import cluster
from sklearn.metrics import mutual_info_score
from tqdm import trange

from estimators.mine import MutualInfoNeuralEstimationNetwork, MutualInfoNeuralEstimationTrainer
from utils.algebra import to_onehot, mutual_info_upperbound
from utils.common import set_seed
from utils.constants import BATCH_SIZE
from utils.simulation import log_softmax_data


class SoftmaxTest:

    def __init__(self, n_samples=20000, n_classes=10, p_argmax=0.99):
        self.n_classes = n_classes
        self.softmax, self.labels = log_softmax_data(n_samples=n_samples, n_classes=n_classes, p_argmax=p_argmax)
        self.mutual_info_true = mutual_info_upperbound(accuracy=p_argmax, n_classes=n_classes)

    def mine(self, noise_variance=0):
        labels = to_onehot(self.labels)
        normal_sampler = torch.distributions.normal.Normal(loc=0, scale=math.sqrt(noise_variance))
        labels = labels.type(torch.float32)
        trainer = MutualInfoNeuralEstimationTrainer(
            MutualInfoNeuralEstimationNetwork(x_size=self.softmax.shape[1], y_size=labels.shape[1]))

        softmax = self.softmax.split(BATCH_SIZE)
        labels = labels.split(BATCH_SIZE)
        n_batches = len(softmax)

        trainer.start_training()
        for epoch in trange(20, desc='Optimizing MINE'):
            for batch_id in random.sample(range(n_batches), k=n_batches):
                labels_batch = labels[batch_id]
                labels_batch += normal_sampler.sample(labels_batch.shape)
                trainer.train_batch(x_batch=softmax[batch_id], y_batch=labels_batch)
        trainer.finish_training()
        print(f"Mutual Information Neural Estimation (MINE) lower-bound: {trainer.get_mutual_info():.3f} "
              f"(true value: {self.mutual_info_true:.3f})")
        plt.plot(np.arange(len(trainer.mutual_info_history)), trainer.mutual_info_history)
        plt.show()

    def kmeans(self, n_clusters=None):
        # n_clusters should be greater or equal to n_classes
        if n_clusters is None:
            n_clusters = self.n_classes
        model = cluster.KMeans(n_clusters=n_clusters, n_jobs=-1)
        softmax_binned = model.fit_predict(self.softmax)
        estimated = mutual_info_score(softmax_binned, self.labels)
        # convert nats to bits
        estimated *= np.log2(np.e)
        print(f"KMeans Mutual Information estimate: {estimated:.3f} (true value: {self.mutual_info_true:.3f})")


if __name__ == '__main__':
    set_seed(26)
    test = SoftmaxTest()
    test.mine()
    test.kmeans()
