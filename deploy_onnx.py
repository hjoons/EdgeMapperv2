import numpy as np
import onnxruntime as rt
import argparse
import time
import torch
import matplotlib.pyplot as plt

from dataloader import get_loader


parser = argparse.ArgumentParser(description='ONNX Deployment code')    

parser.add_argument('--model', type=str,help='Model to be deployed' )
args = parser.parse_args()

model_name = args.model
onnx_model_name = f"{model_name}.onnx" 

print('Loading dataset...')
dataset = get_loader(zipfile='/home/orin/Documents/FH12_23-24/EdgeMapperv2/hw/nyu.zip', batch_size=1, split='eval')


print('Creating inference session...')
# Create Inference session using ONNX runtime
sess_options = rt.SessionOptions()
sess = rt.InferenceSession(onnx_model_name)
sess_options.intra_op_num_threads=4
sess = rt.InferenceSession(onnx_model_name,sess_options)

print('Grabbing input names for ONNX model...')
# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for batch_idx, batch in enumerate(dataset):
    img = batch['image'].numpy()
    
    # Run inference and get the prediction for the input image
    start_time = time.time()
    pred_onnx = sess.run(None, {input_name: img})[0]
    10.0 / pred_onnx
    end_time = time.time()
    print(f'Inference time: {end_time - start_time}')
    
    
    plt.imshow(pred_onnx.squeeze(0).squeeze(0))
    plt.show()



