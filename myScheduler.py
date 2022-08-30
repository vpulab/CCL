from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class myScheduler(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        # >>> # Assuming optimizer uses lr = 0.05 for all groups
        # >>> # lr = 0.05     if epoch < 30
        # >>> # lr = 0.005    if 30 <= epoch < 60
        # >>> # lr = 0.0005   if 60 <= epoch < 90
        # >>> # ...
        # >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        # >>> for epoch in range(100):
        # >>>     train(...)
        # >>>     validate(...)
        # >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, lr, warm_up,  warmup_epochs, warmup_lr, gamma=0.1, max_epochs=9999999, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup = warm_up
        self.max_epochs = max_epochs
        self.ClassFlag = 0
        self.epochs_done = 0

        if self.warmup:
            self.WarmUp_lr_list = np.linspace(warmup_lr, lr, warmup_epochs, endpoint=True)
            self.warmup_epochs = warmup_epochs
            print('Using Warm Up for ' + str(self.warmup_epochs) + ' epochs with initial ' + str(warmup_lr) + ' LR')
        else:
            self.warmup_epochs = 0
            print('NOT using Warm Up')

        super(myScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Scheduler with implemented warm-ip. It only starts to decay the learning rate when all classes have been introduced in
        # the curriculum learning

        # Check if its a warm-up epoch
        if self.warmup and self.last_epoch < self.warmup_epochs:
            print('Epoca de WarmUp')
            new_lr = [self.WarmUp_lr_list[self.last_epoch] for base_lr in self.base_lrs]

        # If not a warmup epoch lr will decay either at max_epochs, or at step_size if ClassFlag = 1
        else:
            if not self.ClassFlag:
                new_lr = [base_lr * self.gamma ** ((self.last_epoch - self.warmup_epochs) // self.max_epochs) for base_lr in self.base_lrs]
                self.epochs_done = self.last_epoch
            else:
                new_lr = [base_lr * self.gamma ** ((self.last_epoch - self.epochs_done) // self.step_size) for base_lr in self.base_lrs]

        return new_lr

    def update_flag(self, ClassFlag):
        self.ClassFlag = ClassFlag


