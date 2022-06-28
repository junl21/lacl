import random
from PIL import Image, ImageFilter

import torch
import torch.utils.data as data


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class CategoryDataset(data.Dataset):
    def __init__(self, data_path, transforms):
        data_file = open(data_path)

        self.transforms = transforms

        self.images = []
        self.labels = []
        try:
            text_lines = data_file.readlines()
            for i in text_lines:
                i = i.strip()
                self.images.append(i.split(' ')[0])
                self.labels.append(int(i.split(' ')[1]))
        finally:
            data_file.close()

    def __getitem__(self, ind):
        image = Image.open(self.images[ind])
        label = self.labels[ind]
        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.images)
