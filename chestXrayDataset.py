
import os
import numpy as np
import torchvision
import torch
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt


class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder: str=""):
        self.image_folder = image_folder
        self.images = os.listdir(image_folder)

    # create one-hot encoding of class index (0 -> [1, 0, 0], 1 -> [0, 1, 0])
    def _get_target(self, class_: int=0):
        target = list(np.zeros((3), dtype=int))
        target[class_] = 1

        return target

    # flip the image with a certain chance
    def _flip(self, image, chance: float=0.5):
        flip = iaa.Fliplr(chance)
        return flip(images=image)

    # get sample
    def __getitem__(self, idx):
        image_file = self.images[idx]

        image = Image.open((self.image_folder + image_file))
        image = np.array(image)
        try:
            image = image[:, :, 0]
        except:
            pass

        image = image / 255

        image = self._flip(image, chance=0.5)
        image = torch.Tensor(image).reshape(1, 512, 512)

        target = int(image_file.split("_")[0])
        target = self._get_target(class_=target)
        target = torch.Tensor(target)

        return image, target

    def __len__(self):
        return len(self.images)
