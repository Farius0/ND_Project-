import math
import os.path

import cv2
import numpy as np
import torch
from skimage import measure, io
from PIL import Image


def get_dice_dict(dice, name):
    columns = [[row[i] for row in dice] for i in range(len(dice[0]))]
    dice_dict = {}
    for c in range(len(columns)):
        dice_dict[name[c]] = columns[c]
    return dice_dict


def get_mad_dict(mad, name):

    mad_dict = {}
    for c in range(len(mad)):
        mad_dict[name[c]] = mad[c]
    return mad_dict


def one_hot(label, num_classes):
    one_hot = np.zeros((num_classes, label.shape[0], label.shape[1]))
    for i in range(num_classes):
        one_hot[i, ...] = (label == i)
    return one_hot


def batch_one_hot(labels, num_classes):
    list = []
    for i in range(labels.shape[0]):
        label = labels[i, ...].squeeze()
        one_hot = np.zeros((num_classes, label.shape[0], label.shape[1]))
        for i in range(num_classes):
            one_hot[i, ...] = (label == i)
        list.append(one_hot)
    one_hot = np.stack(list, axis=0)
    return one_hot


def label_to_contour(label):
    label = np.array(label)
    num_classes = np.max(label) + 1
    output = np.zeros_like(label, dtype=np.uint8)
    for i in range(1, num_classes):
        class_label = (label == i)
        blank_image = np.zeros_like(label)
        contours = measure.find_contours(class_label, 0.8)
        for contour in contours:
            contour = [list(map(int, c)) for c in contour]
            for point in contour:
                x, y = point
                blank_image[x, y] = 255
        boundary = np.zeros_like(label)
        for y in range(blank_image.shape[1]):
            a = np.where(blank_image[:, y])
            if len(a[0]) != 0:
                x = np.min(a[0])
                boundary[x+1, y] = i

        output += boundary

    return output


def batch_label_to_contour(label):

    batch_size = label.shape[0]

    output = np.zeros_like(label)

    for b in range(batch_size):
        output[b] = label_to_contour(label[b])

    return output

def map_to_contour(region, num_contour=9):
    h, w = region.shape
    contour = np.zeros([num_contour, w])
    for l in range(1, num_contour + 1):
        last_row = 0
        for i in range(w):
            vec = region[:, i]
            idxs = np.argwhere(vec == l)
            if len(idxs) != 0:
                idx = np.min(idxs)
            else:
                idx = idxs
            if not np.isscalar(idx):
                idx = last_row
            contour[l-1, i] = idx
            last_row = idx
    return contour

def get_seg_map(b_map, s):

    boundary_num = b_map.shape[0]
    H, W = s.shape
    indexs = torch.arange(0, H).reshape(H, 1).expand(H, W).to(b_map.device)
    masks = torch.zeros((boundary_num + 1, H, W), device=b_map.device)
    masks[0] = indexs < b_map[0]
    for i in range(1, boundary_num):
        masks[i] = (b_map[i-1] <= indexs) & (indexs < b_map[i])
    masks[boundary_num] = indexs >= b_map[boundary_num-1]
    S = torch.zeros_like(s, dtype=torch.float)
    for i in range(1, boundary_num + 1):
        S = torch.maximum(S, masks[i] * i)
    return S


def batch_get_seg_map(b_map, s):
    batch_size = b_map.shape[0]
    S = torch.zeros_like(s)
    for b in range(batch_size):
        S[b] = get_seg_map(b_map[b], s[b])
    return S


def get_layer_name(dataset, num_layer=8):
    layer_name = []
    contour_name = []

    if dataset == "HEGDataset":  # num_layer = 7
        layer_name = ["ILM-GCL[RNFL]", "GCL-INL[GCL+IPL]", "INL-OPL[INL]", "OPL-ONL[OPL]", "ONL-IS[ONL]", "IS-RPE[IS+OS]", "RPE-BM [RPE]"]
        contour_name = ["ILM", "GCL", "INL", "OPL", "ONL", "IS", "RPE", "BM"]
    if dataset == "AierDataset":  # num_layer = 5
        layer_name = ["ILM-INL[RNFL+GCL+IPL]", "INL-ONL[INL+OPL]", "ONL-IS[ONL]", "IS-RPE[IS+OS]", "RPE-BM[RPE]"]
        contour_name = ["ILM", "INL", "ONL", "IS", "RPE", "BM"]
    if dataset == "CellDataset":   # num_layer = 6
        layer_name = ["ILM-INL[RNFL+GCL+IPL]", "INL-OPL[INL]", "OPL-ONL[OPL]", "ONL-IS[ONL]", "IS-RPE[IS+OS]", "RPE-BM[RPE]"]
        contour_name = ["ILM", "INL", "OPL", "ONL", "IS", "RPE", "BM"]

    return layer_name, contour_name


