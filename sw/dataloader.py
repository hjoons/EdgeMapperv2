import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random

_check_pil = lambda x: isinstance(x, Image.Image)

_check_np_img = lambda x: isinstance(x, np.ndarray)

class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=ToTensor(mode))
    
            self.data = DataLoader(self.training_samples,
                                   args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

        elif mode == 'eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=ToTensor(mode))
            self.data = DataLoader(self.testing_samples,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=1)
        else:
            print('mode should be one of \'train\' or \'test\'. Got {}'.format(mode))

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None):
        self.args = args
        if mode == 'eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        if self.mode == 'train':
            rgb_file = sample_path.split()[0]
            depth_file = sample_path.split()[1]

            image_path = os.path.join(self.args.data_path, rgb_file).replace('\\', '/')
            depth_path = os.path.join(self.args.gt_path, depth_file).replace('\\', '/')

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            depth_gt = depth_gt / 1000.0

            image, depth_gt = self.train_preprocess(image, depth_gt)



            sample = {'image': image, 'depth': depth_gt}
        else:
            data_path = self.args.gt_path_eval

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            depth_path = os.path.join(data_path, "./" + sample_path.split()[1])
            
            has_valid_depth = False
            try:
                depth_gt = Image.open(depth_path)
                has_valid_depth = True
            except IOError:
                depth_gt = False
                print('Missing gt for {}'.format(image_path))

            if has_valid_depth:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt = depth_gt / 1000.0

            sample = {'image': image, 'depth': depth_gt}

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
        return len(self.filenames)
    
class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)
        image = self.normalize(image)

        # # put in expected range
        # depth = torch.clamp(depth, 10, 1000)

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