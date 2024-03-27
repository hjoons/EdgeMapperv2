from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from transforms import *

class OTIDataset(Dataset):
    def __init__(self, data, split, transform=None):
        self.data = data
        self.transform = transform
        self.split = split

    def __getitem__(self, idx):
        img = self.data[idx][0]
        depth = self.data[idx][1]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if self.split == 'train':
            if isinstance(depth, np.ndarray):

                depth = (depth / 10000.0 * 255.0).astype('uint8')
                depth = Image.fromarray(depth)

        sample = {'image': img, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)
    
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
        dataset = OTIDataset(data, split, transform=transform)
        return DataLoader(dataset, batch_size, shuffle=True)
    elif split == 'eval':
        transform = eval_transform(resolution)
        dataset = OTIDataset(data, split, transform=transform)
        return DataLoader(dataset, batch_size, shuffle=False)