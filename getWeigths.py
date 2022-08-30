import torch
import numpy as np
import torch.nn.functional as F


def getWeights(CONFIG, Dataset):

    ClassFrequency = np.histogram(Dataset.labelsindex)[0]

    # No weighting
    if CONFIG['DATASET']['WEIGHTING'] == 'None':
        weights = torch.ones(np.max(Dataset.labelsindex) + 1)

    elif CONFIG['DATASET']['WEIGHTING'] == 'Inverse':
        weights = torch.from_numpy(1 / ClassFrequency)

        weights = weights / torch.sum(weights) * int(np.max(Dataset.labelsindex) + 1)

    elif CONFIG['DATASET']['WEIGHTING'] == 'LogInverse':
        weights = torch.from_numpy(1 / np.sqrt(ClassFrequency))

    elif CONFIG['DATASET']['WEIGHTING'] == 'Min':
        weights = torch.from_numpy(np.min(ClassFrequency) / ClassFrequency)

    elif CONFIG['DATASET']['WEIGHTING'] == 'LogMin':
        weights = torch.from_numpy(np.min(ClassFrequency) / np.sqrt(ClassFrequency))

        weights = weights / torch.sum(weights) * int(np.max(Dataset.labelsindex) + 1)

    elif CONFIG['DATASET']['WEIGHTING'] == 'Max':
        weights = torch.from_numpy(np.max(ClassFrequency) / ClassFrequency)

    elif CONFIG['DATASET']['WEIGHTING'] == 'LogMax':
        weights = torch.from_numpy(np.max(ClassFrequency) / np.sqrt(ClassFrequency))

        weights = weights / torch.sum(weights)

    elif CONFIG['DATASET']['WEIGHTING'] == 'EffectiveSamples':
        beta = 0.9999
        samples_per_cls = ClassFrequency
        no_of_classes = int(np.max(Dataset.labelsindex) + 1)

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = torch.from_numpy(weights / np.sum(weights) * no_of_classes)


    # weights = weights / torch.sum(weights)

    return weights.float().cuda()
