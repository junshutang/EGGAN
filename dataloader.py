from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import numpy as np
import os
import utils
import random

# dataloader

class DISFA(data.Dataset):

    def __init__(self, image_dir, attr_path, transform, mode, c_dim):

        self.image_dir = image_dir
        self.attr_path = attr_path
        self.transform = transform
        self.mode = mode
        self.c_dim = c_dim

        self.train_dataset = []
        self.test_dataset = []

        # Fills train_dataset and test_dataset --> [filename, boolean attribute vector]
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

        print("------------------------------------------------")
        print("Training images: ", len(self.train_dataset))
        print("Testing images: ", len(self.test_dataset))

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]

        random.seed(1234)
        random.shuffle(lines)

        # Extract the info from each line
        for idx, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = []  # Vector representing the presence of each attribute in each image

            for n in range(len(values)):
                label.append(float(values[n]))

            if idx < 20000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Dataset ready!...')

    def __getitem__(self, index):

        if self.mode == 'train':
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        return self.num_images


def get_loader(image_dir, attr_path, au_dim, image_size=128,
               batch_size=25, mode='train', num_workers=1):

    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform = T.Compose(transform)

    dataset = DISFA(image_dir, attr_path, transform, mode, au_dim)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)

    return data_loader