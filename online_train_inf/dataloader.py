import cv2
import os

from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from transforms import *
from io import BytesIO

root = 'nyu_data'

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
            # mask = (depth == 0).astype(np.uint8)
            # if np.any(mask):
            #     depth = cv2.inpaint(depth, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            depth = Image.fromarray(depth)


        sample = {'image': img, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)

class TestDataset(Dataset):
    def __init__(self, data, nyu2_split, transform=None):
        self.data, self.nyu_dataset = data, nyu2_split
        self.transform = transform

    def __getitem__(self, idx):

        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[os.path.join(root, sample[0])]))
        depth = Image.open(BytesIO(self.data[os.path.join(root, sample[1])]))
        
        sample = {"image": image, "depth": depth}

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.nyu_dataset)
    
def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}

    nyu2_test = list((row.split(',') for row in (data['nyu_data/data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    print(f'Loaded Test Images: {len(nyu2_test)}).')
    return data, nyu2_test

def get_loader(data, batch_size, resolution, split):
    if split == 'train':
        transform = train_transform(resolution)
        dataset = OTIDataset(data, transform=transform)
        return DataLoader(dataset, batch_size, shuffle=True)
    elif split == 'eval':
        data, nyu2_test = loadZipToMem('./nyu2.zip')
        transform = eval_transform(resolution)
        dataset = TestDataset(data, nyu2_test, transform=transform)
        return DataLoader(dataset, batch_size, shuffle=False)