import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Class definition for the Focal Loss. Extracted from the paper Focal Loss for Dense Object detection by FAIR.
    """

    def __init__(self, focusing_param=1, balance_param=0.25):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, output, target):
        """
        Computes the focal loss for a classification problem (scene classification)
        :param output: Output obtained by the network
        :param target: Ground-truth labels
        :return: Focal loss
        """
        # Compute the regular cross entropy between the output and the target
        logpt = - self.cross_entropy(output, target)
        # Compute pt
        pt = torch.exp(logpt)

        # Compute focal loss
        focal_loss = -((1 - pt) ** self.focusing_param) * logpt
        # Apply weighting factor to obtain balanced focal loss
        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss
