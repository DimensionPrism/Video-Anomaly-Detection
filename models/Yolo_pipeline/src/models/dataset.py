import torch
import torchvision

from torchvision import transforms
from torch.utils.data import Dataset, dataloader

class FER2013(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index]).float()
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)

        return image, label