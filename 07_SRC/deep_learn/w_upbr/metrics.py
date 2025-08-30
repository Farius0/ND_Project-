import numpy as np
import torch
from util import one_hot, batch_one_hot


def get_error(pred, label):
    return np.sum(np.abs(pred-label))

def dice_coefficient(pred, label, epsilon=1e-6):

    d = 0.0

    batch_size = pred.shape[0]

    for i in range(batch_size):

        p_b = pred[i].float().reshape(-1)
        l_b = label[i].float().reshape(-1)

        inter = torch.dot(p_b, l_b)
        union = torch.sum(p_b) + torch.sum(l_b)

        # d += (2.0 * inter + epsilon) / (union + epsilon)

        try:
            dc = 2.0 * inter / union
        except ZeroDivisionError:
            dc = 0.0

        d += dc

    dice = d / batch_size

    return dice

def multi_class_dice_coefficient(pred, label, num_classes, epsilon=1e-6):

    dice_scores = []

    pred = torch.as_tensor(batch_one_hot(pred, num_classes))
    label = torch.as_tensor(batch_one_hot(label, num_classes))

    for c in range(num_classes):
        dice = dice_coefficient(pred[:, c, ...], label[:, c, ...], epsilon)
        dice_scores.append(dice)

    # average_dice = np.mean(dice_scores)

    return dice_scores





















