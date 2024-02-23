import os
import time
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib import image
import cv2
import pyk4a
from pyk4a import PyK4APlayback, Config, FPS, DepthMode, ColorResolution, ImageFormat
import h5py
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import argparse
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from sw.mobilenetv3 import MobileNetSkipConcat
from sw.h5dataset import createH5TestLoader

def compute_errors(gt, pred, epsilon=1e-6):
    """
    Compute error metrics between ground truth and predicted depth maps.

    Args:
        gt (torch.Tensor): Ground truth depth map.
        pred (torch.Tensor): Predicted depth map.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        Tuple: Tuple containing various error metrics.
    """
    # Ensure non-zero and non-negative ground truth values
    gt = gt.float().to('cpu')
    pred = pred.float().to('cpu')

    gt = torch.clamp(gt, min=epsilon)
    pred = torch.clamp(pred, min=epsilon)  # Also ensure predictions are positive

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < (1.25 ** 2)).float().mean()
    a3 = (thresh < (1.25 ** 3)).float().mean()

    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image

# Load an MKV file
def create_frames(vid_path: str, out_file: str):
    # Load an MKV file
    if os.path.exists(f"{vid_path}"):
        mkv_file = f"{vid_path}"
    else:
        print("That path doesn't exist bruh lol")
        exit()

    playback = PyK4APlayback(mkv_file)
    playback.open()

    # print(type(playback))
    # Create a new HDF5 file
    file = h5py.File(f'{out_file}', 'w')

    # Create datasets for RGB images and depth maps
    rgb_images = file.create_dataset('images', (0, 480, 640, 3), maxshape=(None, 480, 640, 3))
    depth_images = file.create_dataset('depths', (0, 480, 640), maxshape=(None, 480, 640))

    i = 0
    while True:
        try:
            capture = playback.get_next_capture()

            if capture.color is not None:
                img_color = cv2.cvtColor(convert_to_bgra_if_required(0, capture.color), cv2.COLOR_BGR2RGB)[120:600, 320:960, 0:3]

                # Append the RGB image to the dataset
                rgb_images.resize(i + 1, axis=0)
                rgb_images[i] = img_color

            if capture.depth is not None:
                img_depth = capture.transformed_depth[120:600, 320:960]

                # Append the depth map to the dataset
                depth_images.resize(i + 1, axis=0)
                depth_images[i] = img_depth

            i += 1

            # key = cv2.waitKey(0)
            # if key != -1:
            #     break
        except EOFError:
            break

    # Close the HDF5 file
    file.close()

    # Close the MKV file
    playback.close()

def inference(model, h5_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load the HDF5 file
    eval_loader = createH5TestLoader(h5_file)
    errors = []

    for i, data in enumerate(eval_loader):
        inputs = data['image'].to(device)
        labels = data['depth'].to(device)

        outputs = model(inputs)

        outputs = outputs.squeeze(0)
        labels = labels.squeeze(0)

        errors.append(compute_errors(labels, outputs))

    error_tensors = [torch.tensor(e).to(device) for e in errors]

    error_stack = torch.stack(error_tensors, dim=0)

    mean_errors = error_stack.mean(0).cpu().numpy()

    abs_rel = mean_errors[0]
    sq_rel = mean_errors[1]
    rmse = mean_errors[2]
    rmse_log = mean_errors[3]
    a1 = mean_errors[4]
    a2 = mean_errors[5]
    a3 = mean_errors[6]
    print(f'abs_rel: {abs_rel}\nsq_rel: {sq_rel}\nrmse: {rmse}\nrmse_log: {rmse_log}\na1: {a1}\na2: {a2}\na3: {a3}\n')



    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_path', type=str, default='data/2021-04-08-15-57-50.mkv')
    parser.add_argument('--out_file', type=str, default='data/2021-04-08-15-57-50.h5')
    parser.add_argument('--model_checkpoint', type=str, default='../mbnv3_epoch_100.pt')

    args = parser.parse_args()

    model = MobileNetSkipConcat()
    model.load_state_dict(torch.load(args.model_checkpoint))

    create_frames(args.vid_path, args.out_file)

    inference(model, args.out_file)

if __name__ == "__main__":
    main()

    



