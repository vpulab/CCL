import torch
from torch import Tensor
import torch.nn as nn
import numpy as np


class SPLLoss(nn.NLLLoss):
    def __init__(self, n_samples=0):
        super(SPLLoss, self).__init__()
        self.n_samples = n_samples
        self.SamplesPercentage = 50
        self.Increase = 10
        self.steplength = 20
        self.v = torch.ones(n_samples).int()

    def forward(self, input, target, index):
        loss = nn.functional.cross_entropy(input, target, reduction="none")
        w_loss = loss * self.v[target].cuda()
        return w_loss

    def increase_classes(self, epoch):
        if (self.SamplesPercentage < 100) and (np.mod(epoch, self.steplength) == 0) and (epoch != 0):
            self.SamplesPercentage += self.Increase

    def update_weigths(self, sample_loss):
        sorted_samples = np.argsort(sample_loss)

        n_samples = int(np.round(self.SamplesPercentage * self.n_samples / 100))

        # Select the classes (number given by ClassesForLoss) to modify the final loss
        in_samples = sorted_samples[0:n_samples]
        out_samples = sorted_samples[n_samples:]

        assert (len(in_samples) + len(out_samples) == len(sorted_samples))

        # Update v
        self.v[out_samples] = 0
        self.v[in_samples] = 1

        return self.v.int()
