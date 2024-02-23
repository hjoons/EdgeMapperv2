import h5py

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import torch


_check_pil = lambda x: isinstance(x, Image.Image)

_check_np_img = lambda x: isinstance(x, np.ndarray)

class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000

        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {"image": image, "depth": depth}

    def to_tensor(self, pic):
        if not (_check_pil(pic) or _check_np_img(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])

def getDefaultTrainTransform():
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomChannelSwap(0.5), ToTensor()]
    )

class H5DepthDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        h5_file = h5py.File(h5_path, 'r')
        self.h5_file = h5_file
        self.transform = transform
        
    def __len__(self):
        return len(self.h5_file['images'])
    
    def __getitem__(self, index):
        img = self.h5_file['images'][index]
        gt = self.h5_file['depths'][index]

        pil_img = Image.fromarray(img, 'RGB')
        pil_gt = Image.fromarray(gt, 'L')

        sample = {"image": pil_img, "depth": pil_gt}
    
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
def createH5TrainLoader(path, batch_size=1):
    transformed_training = H5DepthDataset(path, transform=getDefaultTrainTransform())
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    
    return DataLoader(transformed_training, batch_size=batch_size, shuffle=True), DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)

def createH5TestLoader(path, batch_size=1):
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    return DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)