import numpy
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob

from torchvision.transforms import transforms


class imageNet1kSubDataset(Dataset):
    def __init__(self, data_path_list):
        'Initialization'
        self.data_path_list = data_path_list
        self.data_set_len = len(self.data_path_list)
        # mean and std of imagenet1k
        # source and reason https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)]
        )

    def __len__(self):
        'Denotes the total number of samples'
        return self.data_set_len

    def __getitem__(self, index):
        # Select sample
        path = self.data_path_list[index]
        # Load data and get label
        image = Image.open(path)
        # print(image.format, image.size, image.mode)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        label = path.split('\\')[-3]
        # congrats, you have found my stupid lazy manual one hot encoding.
        if label == 'real':
            label = torch.Tensor([1, 0])
        elif label == 'generated':
            label = torch.Tensor([0, 1])
        else:
            assert False, 'label not found'
        return image, label