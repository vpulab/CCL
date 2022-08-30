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
import matplotlib.pyplot as plt
sys.path.insert(0, './Libs')
sys.path.insert(0, './Libs/Datasets')
import utils
from getConfiguration import getConfiguration
from myScheduler import myScheduler
from CifarDataset import CifarDataset
from getWeigths import getWeights
from FocalLoss import FocalLoss
from ResNet32 import ResNet, BasicBlock
from ClassSPLLoss import ClassSPLLoss
from SPLLoss import SPLLoss

parser = argparse.ArgumentParser(description='Video Classification')
parser.add_argument('--Model', metavar='DIR', help='Folder to be evaluated', required=False)
parser.add_argument('--Dataset', metavar='DIR', help='Dataset to be used', required=False)
parser.add_argument('--Architecture', metavar='DIR', help='Architecture to be used', required=False)
parser.add_argument('--Training', metavar='DIR', help='Training to be used', required=False)
parser.add_argument('--Options', metavar='DIR', nargs='+', help='an integer for the accumulator')


def evaluate(val_loader, model):
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

    loss_function = nn.CrossEntropyLoss(reduction='none')

    val_time_start = time.time()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(val_loader):
            # Start batch_time
            start_time = time.time()
            if USE_CUDA:
                images = mini_batch['Image'].cuda()
                labels = mini_batch['Class'].cuda()

            # Fuse batch size and ncrops to set the input for the network
            bs, ncrops, c_img, h, w = images.size()
            images = images.view(-1, c_img, h, w)

            # CNN Forward
            outputLabels = model(images)

            # Average results over the 10 crops
            outputLabels = outputLabels.view(bs, ncrops, -1).mean(1)

            # Compute and save loss
            loss_per_batch = loss_function(outputLabels, labels.long())
            loss_list.extend(loss_per_batch.cpu())

            loss = torch.mean(loss_per_batch)
            losses.update(loss.item(), batch_size)

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
                print('Validation Batch: [{}/{}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Validation Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                      'Validation Accuracy {accuracy.val:.3f} (avg: {accuracy.avg:.3f})'.
                      format(i, len(val_loader), batch_time=batch_time,
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
        print('Validation results: Accuracy {accuracy.avg:.3f}. Error {error:.3f}'.format(accuracy=accuracy,
                                                                                          error=100 - accuracy.avg))

    return losses, accuracy, CM, class_accuracy, class_loss, pred_list, GT_list


# ----------------------------- #
#   Global Variables & Config   #
# ----------------------------- #

global USE_CUDA, CONFIG
USE_CUDA = torch.cuda.is_available()

args = parser.parse_args()
CONFIG, dataset_CONFIG, architecture_CONFIG, _ = getConfiguration(args)


# ----------------------------- #
#           Networks            #
# ----------------------------- #

# ResNet-32
model = ResNet(BasicBlock, [5, 5, 5], num_classes=CONFIG['DATASET']['N_CLASSES'])

# Extract model parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
model_parameters = sum([np.prod(p.size()) for p in model_parameters])

if USE_CUDA:
    model.cuda()

# Load Model to evaluate
# Model file to load
completePath = os.path.join(args.Model, 'Models', 'ResNet32_' + CONFIG['DATASET']['NAME'] + '_best.pth.tar')

if os.path.isfile(completePath):
    checkpoint = torch.load(completePath)
    best_prec1 = checkpoint['best_prec_val']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print('Loaded model from: ' + completePath + ' with ' + str(best_prec1) + '% of validation accuracy.')
else:
    exit('Model ' + completePath + ' was not found.')


# ----------------------------- #
#           Datasets            #
# ----------------------------- #

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

valDataset = CifarDataset('./Data', train=False, CONFIG=CONFIG, ten_Crop=CONFIG['VALIDATION']['TEN_CROPS'])
val_loader = torch.utils.data.DataLoader(valDataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TEST'], shuffle=False,
                                         num_workers=8, pin_memory=True)

dataset_classes = valDataset.classes


# ----------------------------- #
#          Information          #
# ----------------------------- #

print('Dataset loaded:')
print('Validation set. Size {} video sequences. Batch size {}. Nbatches {}'.
      format(len(val_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['TEST'],
             CONFIG['VALIDATION']['BATCH_SIZE']['TEST'], len(val_loader)))
print('Number of classes: {}' .format(CONFIG['DATASET']['N_CLASSES']))
print('-' * 65)
print('Number of params: {}'. format(model_parameters))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('----------------------------------------------------------------')


# ----------------------------- #
#            Results            #
# ----------------------------- #

losses, accuracy, CM, class_accuracy, class_loss, pred_list, GT_list = evaluate(val_loader, model)

# valDataset = CifarDataset('./Data/Cifar 10', train=False, CONFIG=CONFIG, ten_Crop=False)
# val_loader = torch.utils.data.DataLoader(valDataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
#
# for i, (mini_batch) in enumerate(val_loader):
#     image = mini_batch['Image']
#     prediction = pred_list[i]
#     GT = GT_list[i]
#     class_predicted = valDataset.classes[prediction]
#     class_GT = valDataset.classes[GT]
#
#     if class_GT == 'truck':
#         if prediction != GT:
#             image = np.squeeze(image.numpy())
#
#             # Unnormalize
#             image = utils.unNormalizeImage(image)
#
#             # Plot prediction error
#             plt.figure(1)
#             plt.imshow(image.transpose(1, 2, 0))
#             plt.title('GT: {}. Prediction: {}'.format(class_GT, class_predicted))
#             plt.show()



