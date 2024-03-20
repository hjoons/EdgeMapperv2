import os
import numpy as np
from PIL import Image

channel_sums = np.zeros(3)
channel_sums_sq = np.zeros(3)
num_images = 0

with open('../nyudepthv2_train_files_with_gt_dense.txt', 'r') as f:
    filenames = f.readlines()

    for name in filenames:
        rgb_file = name.split()[0]

        image_path = os.path.join('../sync', rgb_file)
        image = Image.open(image_path)

        image = np.asarray(image, dtype=np.float32) / 255.0

        channel_sums += np.sum(image, axis=(0, 1))
        channel_sums_sq += np.sum(image ** 2, axis=(0, 1))
        num_images += 1

channel_means = channel_sums / (num_images * image.shape[0] * image.shape[1])
channel_stds = np.sqrt((channel_sums_sq / (num_images * image.shape[0] * image.shape[1])) - (channel_means ** 2))

print(f'Over {num_images} images:')
print("Mean RGB values:", channel_means)
print("Standard deviations of RGB values:", channel_stds)