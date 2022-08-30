import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from sklearn.metrics import confusion_matrix
import numpy as np
import shutil
import sys
import torchvision
import yaml
import torchvision.transforms as transforms
sys.path.insert(0, './Libs')
sys.path.insert(0, './Libs/Datasets')
import utils
from getConfiguration import getConfiguration
from myScheduler import myScheduler
from CifarDataset import CifarDataset
from getWeigths import getWeights
from FocalLoss import FocalLoss
from ResNet32 import ResNet, BasicBlock

parser = argparse.ArgumentParser(description='Video Classification')
parser.add_argument('--Dataset', metavar='DIR', help='Dataset to be used', required=False)
parser.add_argument('--Architecture', metavar='DIR', help='Architecture to be used', required=False)
parser.add_argument('--Training', metavar='DIR', help='Training to be used', required=False)
parser.add_argument('--Options', metavar='DIR', nargs='+', help='an integer for the accumulator')


def train(train_loader, model, optimizer, loss_function):

    # Instantiate time metric
    batch_time = utils.AverageMeter()

    # Instantiate loss metric
    losses = utils.AverageMeter()

    # Instantiate precision metric
    accuracy = utils.AverageMeter()

    # Predictions and GT lists
    pred_list = []
    GT_list = []

    # Losses
    loss_list = []
    loss_list_nw = []

    # Switch to train mode
    model.train()

    # Extract batch size
    batch_size = train_loader.batch_size

    loss_function_nw = nn.CrossEntropyLoss(reduction='none')

    train_time_start = time.time()

    for i, (mini_batch) in enumerate(train_loader):
        # Start batch_time
        start_time = time.time()
        if USE_CUDA:
            images = mini_batch['Image'].cuda()
            labels = mini_batch['Class'].cuda()

        # CNN Forward
        outputLabels = model(images)

        # Loss
        loss_per_batch = loss_function(outputLabels, labels.long())
        loss_list.extend(loss_per_batch.cpu())

        loss_per_batch_nw = loss_function_nw(outputLabels, labels.long())
        loss_list_nw.extend(loss_per_batch_nw.cpu())

        loss = torch.mean(loss_per_batch)
        losses.update(loss.item(), batch_size)

        # loss = loss_function(outputLabels, labels.long()) # ONLY FOR FOCAL LOSS

        # Accuracy
        acc = utils.accuracy(outputLabels.data, labels)[0]
        accuracy.update(acc.item(), batch_size)

        # Save predictions
        pred = torch.argmax(outputLabels, dim=1)
        pred_list.extend(pred.cpu())

        # Save Ground-Truth
        GT_list.extend(labels.cpu())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: float(loss))

        batch_time.update(time.time() - start_time)

        if i % CONFIG['TRAINING']['PRINT_FREQ'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                  'Train Accuracy {accuracy.val:.3f} (avg: {accuracy.avg:.3f})'.
                  format(epoch, i, len(train_loader), batch_time=batch_time,
                         loss=losses, accuracy=accuracy))

    # Convert pred_list and GT_list to numpy arrays
    pred_list = torch.stack(pred_list, 0).numpy()
    GT_list = torch.stack(GT_list, 0).numpy()
    loss_list = torch.stack(loss_list, 0).detach().numpy()
    loss_list_nw = torch.stack(loss_list_nw, 0).detach().numpy()

    # Confusion Matrix
    CM = confusion_matrix(GT_list, pred_list)

    # Class Accuracy and Class Loss
    class_accuracy = np.zeros(np.max(GT_list) + 1)
    class_loss = np.zeros(np.max(GT_list) + 1)
    class_loss_nw = np.zeros(np.max(GT_list) + 1)

    for i in range(np.max(GT_list) + 1):
        class_accuracy[i] = CM[i, i] / np.sum(GT_list == i)

        GT_list_indx = (GT_list == i)
        loss_list_indx = loss_list[GT_list_indx]
        loss_class = np.mean(loss_list_indx)
        class_loss[i] = loss_class

        loss_list_indx_nw = loss_list_nw[GT_list_indx]
        loss_class_nw = np.mean(loss_list_indx_nw)
        class_loss_nw[i] = loss_class_nw

    print('Elapsed time for training {time:.3f} seconds'.format(time=time.time() - train_time_start))

    return losses, accuracy, class_accuracy, class_loss, class_loss_nw


def validate(val_loader, model):
    # Instantiate time metric
    batch_time = utils.AverageMeter()

    # Instantiate loss metric
    losses = utils.AverageMeter()

    # Instantiate precision metric
    accuracy = utils.AverageMeter()

    # Predictions and GT lists
    pred_list = []
    GT_list = []

    # Losses
    loss_list = []

    # Switch to eval mode
    model.eval()

    # Extract batch size
    batch_size = val_loader.batch_size

    loss_function_val = nn.CrossEntropyLoss(reduction='none')

    val_time_start = time.time()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(val_loader):
            # Start batch_time
            start_time = time.time()
            if USE_CUDA:
                images = mini_batch['Image'].cuda()
                labels = mini_batch['Class'].cuda()

            # CNN Forward
            outputLabels = model(images)

            # Compute and save loss
            loss_per_batch = loss_function_val(outputLabels, labels.long())
            loss_list.extend(loss_per_batch.cpu())

            loss = torch.mean(loss_per_batch)
            losses.update(loss.item(), batch_size)

            # loss = loss_function(outputLabels, labels.long())  # ONLY FOR FOCAL LOSS

            # Compute and save accuracy
            acc = utils.accuracy(outputLabels.data, labels)
            accuracy.update(acc[0].item(), batch_size)

            # Save predictions
            pred = torch.argmax(outputLabels, dim=1)
            pred_list.extend(pred.cpu())

            # Save Ground-Truth
            GT_list.extend(labels.cpu())

            batch_time.update(time.time() - start_time)

            if i % CONFIG['TRAINING']['PRINT_FREQ'] == 0:
                print('Validation Batch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Validation Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                      'Validation Accuracy {accuracy.val:.3f} (avg: {accuracy.avg:.3f})'.
                      format(epoch, i, len(val_loader), batch_time=batch_time,
                             loss=losses, accuracy=accuracy))

        # Convert pred_list and GT_list to numpy arrays
        pred_list = torch.stack(pred_list, 0).numpy()
        GT_list = torch.stack(GT_list, 0).numpy()
        loss_list = torch.stack(loss_list, 0).detach().numpy()

        # Confusion Matrix
        CM = confusion_matrix(GT_list, pred_list)

        # Class Accuracy and Class Loss
        class_accuracy = np.zeros(np.max(GT_list) + 1)
        class_loss = np.zeros(np.max(GT_list) + 1)

        for i in range(np.max(GT_list) + 1):
            class_accuracy[i] = CM[i, i] / np.sum(GT_list == i)

            GT_list_indx = (GT_list == i)
            loss_list_indx = loss_list[GT_list_indx]
            loss_class = np.mean(loss_list_indx)
            class_loss[i] = loss_class

        print('Elapsed time for evaluation {time:.3f} seconds'.format(time=time.time() - val_time_start))
        print('Validation results: Accuracy {accuracy.avg:.3f}'.format(accuracy=accuracy))

    return losses, accuracy, CM, class_accuracy, class_loss


# ----------------------------- #
#   Global Variables & Config   #
# ----------------------------- #

global USE_CUDA, CONFIG
USE_CUDA = torch.cuda.is_available()

args = parser.parse_args()
CONFIG, dataset_CONFIG, architecture_CONFIG, training_CONFIG = getConfiguration(args)

print('The following configuration is used for the training')
print(yaml.dump(CONFIG, allow_unicode=True, default_flow_style=False))

# exit()

# Initialize best precision
best_prec = 0

print('Training starts.')
print('-' * 65)


# ----------------------------- #
#         Results Folder        #
# ----------------------------- #

# Create folders to save results
Date = str(time.localtime().tm_year) + '-' + str(time.localtime().tm_mon).zfill(2) + '-' + str(time.localtime().tm_mday).zfill(2) +\
       ' ' + str(time.localtime().tm_hour).zfill(2) + ':' + str(time.localtime().tm_min).zfill(2) + ':' + str(time.localtime().tm_sec).zfill(2)
ResultsPath = os.path.join(CONFIG['MODEL']['OUTPUT_DIR'], Date + ' ' + CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'])

os.mkdir(ResultsPath)
os.mkdir(os.path.join(ResultsPath, 'Images'))
os.mkdir(os.path.join(ResultsPath, 'Images', 'Dataset'))
os.mkdir(os.path.join(ResultsPath, 'Files'))
os.mkdir(os.path.join(ResultsPath, 'Models'))


# Copy files to result folder
shutil.copyfile('trainCNNs.py', os.path.join(ResultsPath, 'trainCNNs.py'))

with open(os.path.join(ResultsPath, 'config_' + args.Dataset + '.yaml'), 'w') as file:
    yaml.safe_dump(dataset_CONFIG, file)
with open(os.path.join(ResultsPath, 'config_' + args.Architecture + '.yaml'), 'w') as file:
    yaml.safe_dump(architecture_CONFIG, file)
with open(os.path.join(ResultsPath, 'config_' + args.Training + '.yaml'), 'w') as file:
    yaml.safe_dump(training_CONFIG, file)


# ----------------------------- #
#           Networks            #
# ----------------------------- #

# Given the configuration file build the desired CNN network
# model = torchvision.models.resnet18(num_classes=CONFIG['DATASET']['N_CLASSES'], pretrained=False)

# ResNet-32
model = ResNet(BasicBlock, [5, 5, 5], num_classes=CONFIG['DATASET']['N_CLASSES'])

# Extract model parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
model_parameters = sum([np.prod(p.size()) for p in model_parameters])

if USE_CUDA:
    model.cuda()


# ----------------------------- #
#           Datasets            #
# ----------------------------- #

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

trainDataset = CifarDataset('./Data/Cifar 10', train=True, CONFIG=CONFIG, imbalance_factor=CONFIG['DATASET']['IMBALANCE'])
train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'], shuffle=True,
                                           num_workers=8, pin_memory=True)

valDataset = CifarDataset('./Data/Cifar 10', train=False, CONFIG=CONFIG)
val_loader = torch.utils.data.DataLoader(valDataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TEST'], shuffle=False,
                                         num_workers=8, pin_memory=True)

dataset_classes = trainDataset.classes

# ----------------------------- #
#          Information          #
# ----------------------------- #

print('Dataset loaded:')
print('Train set. Size {} video sequences. Batch size {}. Nbatches {}'.format(len(train_loader) * CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'],
                                                                              CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'], len(train_loader)))
print('Validation set. Size {} video sequences. Batch size {}. Nbatches {}'.format(len(val_loader) * CONFIG['TRAINING']['BATCH_SIZE']['TEST'],
                                                                                   CONFIG['TRAINING']['BATCH_SIZE']['TEST'], len(val_loader)))
print('Number of classes: {}' .format(CONFIG['DATASET']['N_CLASSES']))
print('-' * 65)
print('Number of params: {}'. format(model_parameters))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('----------------------------------------------------------------')

print(model)

utils.saveBatchExample(train_loader, os.path.join(ResultsPath, 'Images', 'Dataset', 'Training Batch Sample.png'))
utils.saveBatchExample(val_loader, os.path.join(ResultsPath, 'Images', 'Dataset', 'Validation Batch Sample.png'))

utils.plotDatasetHistograms(trainDataset, os.path.join(ResultsPath, 'Images', 'Dataset'), dataset_classes, set='Training', save=True)
utils.plotDatasetHistograms(valDataset, os.path.join(ResultsPath, 'Images', 'Dataset'), dataset_classes, set='Validation', save=True)

# ----------------------------- #
#        Hyper Parameters       #
# ----------------------------- #

# Optimizers
if CONFIG['TRAINING']['OPTIMIZER']['NAME'] == 'SGD':
    # Stochastic Gradient Descent
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), CONFIG['TRAINING']['OPTIMIZER']['LR'],
                                momentum=CONFIG['TRAINING']['OPTIMIZER']['MOMENTUM'], weight_decay=CONFIG['TRAINING']['OPTIMIZER']['WEIGHT_DECAY'])

    if CONFIG['TRAINING']['WARMUP']['ENABLE']:
        print('Using Warm Up for ' + str(CONFIG['TRAINING']['WARMUP']['EPOCHS']) + ' epochs with initial ' + str(CONFIG['TRAINING']['WARMUP']['LR']) + ' LR')

        scheduler = myScheduler(optimizer, CONFIG['TRAINING']['OPTIMIZER']['LR_DECAY'], CONFIG['TRAINING']['OPTIMIZER']['LR'],
                                CONFIG['TRAINING']['WARMUP']['EPOCHS'], CONFIG['TRAINING']['WARMUP']['LR'], CONFIG['TRAINING']['OPTIMIZER']['GAMMA'])
    else:
        # Learning rate decay
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CONFIG['TRAINING']['OPTIMIZER']['LR_DECAY'], CONFIG['TRAINING']['OPTIMIZER']['GAMMA'])


else:
    raise Exception('Optimizer {} was indicate in {} file. This optimizer is not supported.\n'
                    'The following optimizers are supported: SGD'
                    .format(CONFIG['TRAINING']['OPTIMIZER']['NAME'], args.ConfigPath))


# Weighting
weights = getWeights(CONFIG=CONFIG, Dataset=trainDataset)


# Loss Functions
if CONFIG['TRAINING']['LOSS']['NAME'] == 'CROSS ENTROPY':
    loss_function = nn.CrossEntropyLoss(weight=weights, reduction='none')
elif CONFIG['TRAINING']['LOSS']['NAME'] == 'FOCAL LOSS':
    loss_function = FocalLoss()
elif CONFIG['TRAINING']['LOSS']['NAME'] == 'CB LOSS':
    loss_function = nn.CrossEntropyLoss(weight=weights, reduction='none')


# ----------------------------- #
#           Training            #
# ----------------------------- #

# Metrics per epoch
train_loss_list = []
val_loss_list = []
train_accuracy_list = []
val_accuracy_list = []

# List to plot standard deviation
train_loss_list_up = []
train_loss_list_low = []
val_loss_list_up = []
val_loss_list_low = []

# List to plot Learning Rate
lr_list = []

# Weight List
weight_list = []

# Train Class Accuracy List
train_Class_accuracy_list = []
val_Class_accuracy_list = []

# Train Class Loss List
train_Class_loss_list = []
train_Class_loss_list_nw = []
val_Class_loss_list = []

for epoch in range(CONFIG['TRAINING']['EPOCHS']):
        # Epoch time start
        epoch_start = time.time()

        lr_list.append(optimizer.param_groups[0]['lr'])
        weight_list.append(torch.unsqueeze(weights.cpu(), dim=0))

        # Train one epoch
        train_loss, train_accuracy, train_Class_Accuracy, train_Class_Loss, train_Class_Loss_nw = train(train_loader, model, optimizer, loss_function)

        # Validate one epoch
        val_loss, val_accuracy, CM, val_Class_Accuracy, val_Class_Loss = validate(val_loader, model)

        scheduler.step()

        a = np.sum(train_Class_Loss_nw)
        # If Weight Update then update weigths
        # if CONFIG['DATASET']['WEIGHT_UPDATE'] and CONFIG['DATASET']['WEIGHTING'] != 'None':
        if epoch > CONFIG['DATASET']['WEIGHT_FREZEE_EPOCHS'] and CONFIG['DATASET']['WEIGHT_UPDATE']:
            Tao = 1
            alfa = 2
            for i in range(weights.shape[0]):
                # Metodo con precisiones
                # term1 = 1 + alfa * (1 - (min(Tao, train_Class_Accuracy[i]) / Tao))
                # weights[i] = weights[i] * term1

                # Metodo con losses
                # term2 = 1 + train_Class_Loss2[i] / (np.min(train_Class_Loss2) + 10e-20)
                # weights[i] = weights[i] * term2

                # term3 = 1 + train_Class_Loss_nw[i]
                # weights[i] = weights[i] * term3

                term4 = 1 + train_Class_Loss_nw[i] / np.sum(train_Class_Loss_nw)
                weights[i] = weights[i] * term4

            weights = weights / torch.sum(weights) * weights.shape[0]
            loss_function = nn.CrossEntropyLoss(weight=weights, reduction='none')

        # Save Epoch Losses Mean
        train_loss_list.append(train_loss.avg)
        val_loss_list.append(val_loss.avg)

        # Save Epoch Losses STD
        train_loss_list_up.append(train_loss.avg + train_loss.std)
        train_loss_list_low.append(train_loss.avg - train_loss.std)
        val_loss_list_up.append(val_loss.avg + val_loss.std)
        val_loss_list_low.append(val_loss.avg - val_loss.std)

        # Save Epoch Accuracies
        train_accuracy_list.append(train_accuracy.avg)
        val_accuracy_list.append(val_accuracy.avg)

        # Save Epoch Class Accuracies
        train_Class_accuracy_list.append(np.expand_dims(train_Class_Accuracy, axis=0))
        val_Class_accuracy_list.append(np.expand_dims(val_Class_Accuracy, axis=0))

        # Save Epoch Class Losses
        train_Class_loss_list.append(np.expand_dims(train_Class_Loss, axis=0))
        train_Class_loss_list_nw.append(np.expand_dims(train_Class_Loss_nw, axis=0))
        val_Class_loss_list.append(np.expand_dims(val_Class_Loss, axis=0))

        # Plot all the results
        utils.plotTrainingResults(train_loss_list, val_loss_list, train_loss_list_low, train_loss_list_up, val_loss_list_low, val_loss_list_up,
                                  train_accuracy_list, val_accuracy_list, lr_list, weight_list, train_Class_accuracy_list, val_Class_accuracy_list,
                                  train_Class_loss_list, train_Class_loss_list_nw, val_Class_loss_list, ResultsPath, CONFIG, dataset_classes)

        # Epoch time
        epoch_time = (time.time() - epoch_start) / 60

        # Save model
        is_best = val_accuracy.avg > best_prec
        best_prec = max(val_accuracy.avg, best_prec)
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'CONFIG': CONFIG,
            'state_dict': model.state_dict(),
            'best_prec_train': train_accuracy.avg,
            'best_prec_val': val_accuracy.avg,
            'time_per_epoch': epoch_time,
            'model_parameters': model_parameters,
            'confusion_matrix': CM,
            'class_accuracy': val_Class_Accuracy,
            'weights': weights.cpu()
        }, is_best, ResultsPath, dataset_classes, CONFIG['MODEL']['ARCH'] + '_' + CONFIG['DATASET']['NAME'])

        print('Elapsed time for epoch {}: {time:.3f} minutes'.format(epoch, time=epoch_time))
