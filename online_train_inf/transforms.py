import numpy as np
import torch
from torchvision import transforms, utils
from PIL import Image
import random
from itertools import permutations

_check_pil = lambda x: isinstance(x, Image.Image)

_check_np_img = lambda x: isinstance(x, np.ndarray)

class ToTensor(object):
    def __init__(self, test=False, maxDepth=1000.0):
        self.test = test
        self.maxDepth = maxDepth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        transformation = transforms.ToTensor()
        image = np.array(image).astype(np.float32) / 255.0
        depth = np.array(depth).astype(np.float32) # 0-255.0

        if self.test:
            """
            If test, move image to [0,1] and depth to [0, 1]

            ***I don't understand why we do this***
            """
            depth = depth / 1000.0
            image, depth = transformation(image), transformation(depth)
        else:
            depth = depth / 255.0 * 10.0

            zero_mask = depth == 0.0
            image, depth = transformation(image), transformation(depth)

            depth = torch.clamp(depth, self.maxDepth/100.0, self.maxDepth) 
            depth = self.maxDepth / depth

            depth[:, zero_mask] = 0.0
        

        # print('Depth after, min: {} max: {}'.format(depth.min(), depth.max()))
        # print('Image, min: {} max: {}'.format(image.min(), image.max()))

        image = torch.clamp(image, 0.0, 1.0)
        return {'image': image, 'depth': depth}


class RandomHorizontalFlip(object):
    def __call__(self, sample):

        img, depth = sample["image"], sample["depth"]

        if not _check_pil(img):
            raise TypeError("Expected PIL type. Got {}".format(type(img)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img, "depth": depth}


class RandomChannelSwap(object):
    def __init__(self, probability):

        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):

        image, depth = sample["image"], sample["depth"]

        if not _check_pil(image):
            raise TypeError("Expected PIL type. Got {}".format(type(image)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(
                image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])]
            )

        return {"image": image, "depth": depth}

def train_transform():
    transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(.25),
        ToTensor(test=False, maxDepth=10.0)
    ])
    return transform

def eval_transform():
    return ToTensor(test=True, maxDepth=10.0)