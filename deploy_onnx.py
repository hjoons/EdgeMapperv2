import numpy as np
import onnxruntime as rt
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time

# TODO: create argument parser object
parser = argparse.ArgumentParser(description='ECE361E HW3 - Deployment code')
# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model', type=str, help='Model that will be used')
parser.add_argument('--device', type=str,help='Deployment device' )
# TODO: Modify the rest of the code to use those arguments correspondingly
args = parser.parse_args()
device_name = args.device
model_name = args.model
onnx_model_name = f"{device_name}_{model_name}.onnx"  # TODO: insert ONNX model name, essentially the path to the onnx model
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
# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
test_correct = 0
test_total = 0
start_time = time.time()
# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("/home/student/HW3_files/test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("/home/student/HW3_files/test_deployment", filename)).resize((32, 32)) as img:
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
        
        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])
        # Get the label of the predicted class
        pred_class = label_names[top_prediction]
                
        # TODO: compute test accuracy of the model
        gt = filename.replace('.png', '').split('_')[1]
                
        test_total += 1
        test_correct += 1 if pred_class == gt else 0
end_time = time.time()
print('Test accuracy: %.2f %%' % (100. * test_correct / test_total))
print('Total inference time:', (end_time - start_time))