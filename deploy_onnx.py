import numpy as np
import onnxruntime as rt
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from zipfile import ZipFile
from io import BytesIO

class NYU_Testset(Dataset):
    def __init__(self, zip_path):
        input_zip=ZipFile(zip_path)
        data = {name: input_zip.read(name) for name in input_zip.namelist()}
        
        self.rgb = np.load(BytesIO(data['eigen_test_rgb.npy'])).type(torch.float32) #Range [0,1]
        self.depth = np.load(BytesIO(data['eigen_test_depth.npy'])).type(torch.float32) #Range[0, 10]

    def __getitem__(self, idx):
        image = self.rgb[idx]
        depth = self.depth[idx]
        return image, depth

    def __len__(self):
        return len(self.rgb)
    

parser = argparse.ArgumentParser(description='ONNX Deployment code')    

parser.add_argument('--model', type=str,help='Model to be deployed' )
args = parser.parse_args()

model_name = args.model
onnx_model_name = f"{model_name}.onnx" 

# Create Inference session using ONNX runtime
sess_options = rt.SessionOptions()
sess = rt.InferenceSession(onnx_model_name)
sess_options.intra_op_num_threads=4
sess = rt.InferenceSession(onnx_model_name,sess_options)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# start_time = time.time()

dataset = NYU_Testset("/home/orin/Documents/FH12_23-24/EdgeMapperv2/hw/nyu.zip")


# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for image in dataset.depth:
    # Take each image, one by one, and make inference
    with Image.open(image).resize((32, 32)) as img:
        # print("Image shape:", np.float32(img).shape)
        # normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        # Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)
        # change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])
        # print("Input Image shape:", input_image.shape)
        # Run inference and get the prediction for the input image
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        # Remove Batch axis
        pred_image = np.squeeze(pred_onnx)

# end_time = time.time()