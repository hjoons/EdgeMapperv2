import torch
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transforms import *

class OTIDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx][0]
        depth = self.data[idx][1]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(depth, np.ndarray):
            depth = (depth / 10000.0 * 255.0).astype('uint8')
            mask = np.where(depth == 0, 0, 255).astype('uint8')
            depth = cv2.inpaint(depth, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            depth = Image.fromarray(depth)


        # sample = {'image': pil_img, 'depth': pil_depth}
        sample = {'image': img, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)

def get_loader(data, batch_size):
    transform = train_transform()
    dataset = OTIDataset(data, transform=transform)
    return DataLoader(dataset, batch_size, shuffle=True)