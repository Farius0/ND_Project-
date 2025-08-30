import os, numpy as np, torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data_augmentation import build_transforms
import torchvision.transforms.functional as F
import data_augmentation as T


# OCT
class OCTDataset(Dataset):
    """
    root: root path
    num_classes: class number
    mode: train or test
    sup_type: Full or Scribble
    transform: transform operations
    """

    def __init__(self,
                 root,
                 num_classes,
                 mode="train",
                 sup_type="Full",
                 transform=None):

        self.root = root
        self.num_classes = num_classes
        self.mode = mode
        self.sup_type = sup_type
        self.transform = transform

        # train or test
        assert self.mode in ["train", "test", "val"], "Mode ['{}'] is incorrect, please re-enter!".format(self.mode)

        # load train: images labels scribbles
        if self.mode == "train":
            self.image_path = os.path.join(self.root, self.mode, "images")
            if self.sup_type == "Full":
                self.label_path = os.path.join(self.root, self.mode, "labels")
            if self.sup_type == "Scribble":
                self.label_path = os.path.join(self.root, self.mode, "scribbles")
        # load test or val: images labels
        if self.mode == "test" or self.mode == "val":
            self.image_path = os.path.join(self.root, self.mode, "images")
            self.label_path = os.path.join(self.root, self.mode, "labels")

        self.data_list = [file for file in os.listdir(self.image_path)]

    def __len__(self):
        print("the mode is: {}, the supervision type is: {}, the length of the data list is: {}".format(self.mode, self.sup_type, len(self.data_list)))
        return len(self.data_list)

    def __getitem__(self, item):

        filename = self.data_list[item]
        image_path = os.path.join(self.image_path, filename)
        label_path = os.path.join(self.label_path, filename)

        image = Image.open(image_path)
        label = Image.open(label_path)

        sample = {"image": image, "label": label}

        if self.mode == "train":
            H, W = 256, 256
            trans = [
                T.RandomCrop((H, W), pad_if_needed=True),
                T.RandomFlip(p=0.5, direction='horizontal'),
                T.ToTensor(),
            ]
            sample = build_transforms(trans)(sample)

            image, label = sample["image"], sample["label"]

        if self.mode == "test" or self.mode == "val":
            sample = build_transforms([
                T.ToTensor(),
            ])(sample)

            image, label = sample["image"], sample["label"]

        return {'image': image, 'label': label, 'filename': filename.split(".")[0]}





