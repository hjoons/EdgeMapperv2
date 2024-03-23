import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transforms import *

class OTIDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        to_image = transforms.ToPILImage()
        to_depth = transforms.ToPILImage()
        
        img = self.data[idx][0]
        depth = self.data[idx][1]

        pil_img = to_image(img)
        pil_depth = to_depth(depth)

        sample = {'image': pil_img, 'depth': pil_depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)

def get_loader(data, batch_size):
    transform = train_transform()
    dataset = OTIDataset(data, transform=transform)
    return DataLoader(dataset, batch_size, shuffle=True)