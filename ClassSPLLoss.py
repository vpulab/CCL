import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from FocalLoss import FocalLoss


class ClassSPLLoss(nn.NLLLoss):
    def __init__(self, CONFIG, sorted_classes=None):
        super(ClassSPLLoss, self).__init__()
        self.n_classes = CONFIG['DATASET']['N_CLASSES']

        # Curriculum parameters are in percentage. Convert them to classes
        self.ClassesForLoss = round(CONFIG['TRAINING']['LOSS']['START_CLASSES'] * self.n_classes / 100)
        self.Increment = round(CONFIG['TRAINING']['LOSS']['INCREMENT'] * self.n_classes / 100)
        # Epochs of step
        self.step = CONFIG['TRAINING']['LOSS']['STEP']

        # Vector of sorted classes
        if not CONFIG['TRAINING']['LOSS']['ANTICURRICULUM']:
            # Curriculum Learning
            self.sorted_classes = sorted_classes
        else:
            # Anti-Curriculum Learning
            self.sorted_classes = np.flipud(sorted_classes).copy()

        # Vector for loss masking
        self.v = torch.ones(CONFIG['DATASET']['N_CLASSES']).int()

        # Define loss
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        if sorted_classes is None:
            self.self_paced = True
        else:
            self.self_paced = False

    def forward(self, pred, target):
        # Compute loss
        loss = self.criterion(pred, target)

        # Mask loss depending on the classes defined by the curriculum
        w_loss = loss * self.v[target].cuda()
        return w_loss

    def mixup_forward(self, pred, y_a, y_b, lam):
        # Compute loss
        loss = lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

        return loss

    def increase_classes(self, epoch):
        # Regular increase of classes for the loss
        if (self.ClassesForLoss < self.n_classes) and (np.mod(epoch, self.step) == 0) and (epoch != 0):
            self.ClassesForLoss += self.Increment

    def increase_classes_precision_based(self, train_Class_Accuracy):
        # Function that given a set of classes accuracies sets the number of classes that will be used in the loss

        # Get min class accuracy
        acc_min = np.min(train_Class_Accuracy[self.in_classes])

        if (self.ClassesForLoss < self.n_classes) and (acc_min >= 0.9):
            self.ClassesForLoss += self.Increment

    def update_curriculum(self, class_loss):
        # Function to selected the number of classes that will be taken into account for the loss
        # Only update class order based on class loss if self_paced
        if self.self_paced:
            # Sort Class depending on the class loss
            self.sorted_classes = np.argsort(class_loss)

        # Select the classes (number given by ClassesForLoss) to modify the final loss
        in_classes = self.sorted_classes[0:self.ClassesForLoss]
        out_classes = self.sorted_classes[self.ClassesForLoss:]

        # Update v
        self.v[out_classes] = 0
        self.v[in_classes] = 1

    def all_classes_in(self):
        return self.ClassesForLoss == self.n_classes
