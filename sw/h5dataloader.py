import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import random

_check_pil = lambda x: isinstance(x, Image.Image)

_check_np_img = lambda x: isinstance(x, np.ndarray)

class NewH5DataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadH5Preprocess(args, mode, transform=ToTensor())
    
            self.data = DataLoader(self.training_samples,
                                   args.batch_size,
                                   shuffle=True,
                                   num_workers=4)

        elif mode == 'eval':
            self.testing_samples = DataLoadH5Preprocess(args, mode, transform=ToTensor())
            self.data = DataLoader(self.testing_samples,
                                   args.batch_size,
                                   shuffle=False,
                                   num_workers=1)
        else:
            print('mode should be one of \'train\' or \'eval\'. Got {}'.format(mode))
    
    def __len__(self):
        return len(self.data)

class DataLoadH5Preprocess(Dataset):
    def __init__(self, args, mode, transform=None):
        self.args = args
        self.mode = mode
        self.transform = transform
        if mode == 'train':
            self.h5_file = h5py.File(args.train_path, 'r')
        elif mode == 'eval':
            self.h5_file = h5py.File(args.test_path, 'r')
        else:
            print('mode should be one of \'train\' or \'eval\'. Got {}'.format(mode))
    
    def __getitem__(self, idx):
        img = self.h5_file['images'][idx]
        gt = self.h5_file['depths'][idx]
        
        pil_img = img / 255.0
        pil_gt = gt / 1000.0
        
        # pil_img = Image.fromarray(img, 'RGB')
        # pil_gt = Image.fromarray(gt, 'L')

        # pil_img = np.asarray(pil_img, dtype=np.float32) / 255.0            
        # pil_gt = np.asarray(pil_gt, dtype=np.float32) / 1000.0
        
        pil_gt = np.expand_dims(pil_gt, axis=0) 

        if self.mode == 'train':
            pil_img, pil_gt = self.train_preprocess(pil_img, pil_gt)

        sample = {'image': pil_img, 'depth': pil_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.h5_file['images'])
    
class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.488, 0.418, 0.401], std=[0.289, 0.296, 0.308])

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)
        image = self.normalize(image)

        # put in expected range
        # depth = torch.clamp(depth, 10, 1000)

        return {"image": image, "depth": depth}

    def to_tensor(self, pic):
        if not (_check_pil(pic) or _check_np_img(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float()

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
            return img.float()
        else:
            return img
