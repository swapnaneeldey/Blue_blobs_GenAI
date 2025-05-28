# File: astro_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class AstroDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (numpy array): Array of images (N, H, W, C), values expected in [0.0, 1.0].
            labels (numpy array): Array of labels (N,).
            transform (callable, optional): Optional transform to be applied.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Image is already float32, HWC, [0, 1]
        image = self.images[idx]
        label = self.labels[idx]

        # transforms.ToTensor() will handle HWC->CHW and numpy->tensor
        if self.transform:
             image = self.transform(image)
        else:
             # If no transform (unlikely here), manually convert
             image = torch.from_numpy(image.transpose((2, 0, 1))) # CHW format

        label = torch.tensor(label, dtype=torch.long)
        return image, label