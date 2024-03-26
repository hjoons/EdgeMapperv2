import threading
import queue
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time
from torch import nn
import torch.nn.functional as F
import os
from mobilenetv3 import MobileNetSkipConcat
from GuideDepth.model.loader import load_model
import matplotlib.pyplot as plt
import pyk4a
from pyk4a import PyK4A, Config, FPS, DepthMode, ColorResolution
import cv2
from dataloader import get_loader
import pickle as pkl
from loss import Depth_Loss
import numpy as np

def inference_thread():
    fps = 16
    time_per_frame = 1 / fps  # 0.041666666666666664 = 41.6ms
    # Pre-trained weights
    model_path = "net.pth"

    # # Assuming you are not loading a pruned PyTorch model
    # modelI = load_model('GuideDepth').to(device)
    # modelI.eval()
    # modelI.reload_weights()

    # Assuming you are using a pruned PyTorch model
    modelI = torch.load(model_path)
    modelI.eval()
    
    pred_frames = 0

    print(f"Starting capture...")
    config = Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_15
        #color_format=pyk4a.ImageFormat.COLOR_BGRA32,
    )

    k4a = PyK4A(config)
    height = 240
    width = 320
    start_height = (720 - height) // 2
    start_width = (1280 - width) // 2
    end_height = start_height + height
    end_width = start_width + width

    # Open the device
    k4a.start()

    while True:
        capture = k4a.get_capture()
        start_time_all = time.time()
        if stop_event.is_set():
            break

        if weight_buffer.full():
            print('Updating model...')
            # Assuming we are working with a pruned model
            modelI = weight_buffer.get()
            modelI.eval()

        if capture is not None:
            # Get the color image from the capture
            color_image = capture.color

            # transformed_depth_image purely for display
            transformed_depth_image = capture.transformed_depth

            transformed_depth_image = transformed_depth_image[start_height:end_height, start_width:end_width] / 10000.0
            # zeros = transformed_depth_image == 0.0
            # transformed_depth_image[zeros] = 0.9
            # print(f"Numpy image min: {np.min(transformed_depth_image)}")
            
            transformed_depth_image_int8 = (transformed_depth_image * 255).astype(np.uint8)        
            transformed_depth_image_int8 = cv2.applyColorMap(transformed_depth_image_int8, cv2.COLORMAP_JET)
            
            # depth tensor used in training
            # depth_tensor = torch.from_numpy(capture.transformed_depth[start_height:end_height, start_width:end_width])
            # depth_tensor.unsqueeze(0)

            depth_np = capture.transformed_depth[start_height:end_height, start_width:end_width]
            color_np = color_image[start_height:end_height, start_width:end_width, 0:3]
                        
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)[start_height:end_height, start_width:end_width, 0:3]
            color_image_tensor = torch.from_numpy(color_image_rgb)
            color_image_tensor = color_image_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            color_image_tensor = color_image_tensor.float().div(255)

            # Put the (image, label) tuple in the buffer
            if buffer.qsize() >= max_buffer_size:
                # print("Removed the oldest entry from the buffer.")
                # data = [buffer.get() for _ in range(max_buffer_size)]
                # with open('buffer.pickle', 'wb') as f:
                #     pkl.dump(data, f)
                # exit(-1)

                # Remove the oldest item to make space for the new one
                buffer.get()  # This line removes the oldest entry from the queue

            buffer.put((color_np, depth_np))  # Move image back to CPU before storing in buffer

            # print("Added a new entry to the buffer.")

            model_input = color_image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                pred_start = time.time()
                pred = modelI(model_input)
                pred_end = time.time()

                # if pred_frames > 25:
                #     pred_time.append((pred_end - pred_start))

                pred = pred.detach().squeeze(0).squeeze(0).cpu().numpy()

                zero_mask = pred == 0.0
                
                pred = np.clip(pred, 10.0/100.0, 10.0) 
                pred = 10.0 / pred

                pred[zero_mask] = 0.0

                pred = ((pred / 10.0) * 255).astype(np.uint8)
                pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            
            cv2.imshow("Real-Time Video", color_image_rgb)
            cv2.imshow("Depth Image", transformed_depth_image_int8)
            cv2.imshow("Prediction", pred)
            time.sleep(max(0., time_per_frame - (time.time() - start_time_all)))
            

            # Evaluating latency metrics
            # if pred_frames < 1000:
            #     with open('inference_time.csv', 'a+') as f:
            #         f.write(f"{pred_end - pred_start}\n")
            # else:
            #     print('Reached 1000 frames')

            pred_frames += 1

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    # Stop the cameras and close the device
    k4a.device_stop_cameras()
    k4a.device_close()

    # Close the OpenCV window
    cv2.destroyAllWindows()

def training_thread():
    # # Assuming that you are not using a pruned PyTorch model    
    # modelT = load_model('GuideDepth').to(device)
    # modelT.train()
    # modelT.reload_weights()

    # Assuming that you are using a pruned PyTorch model
    modelT = torch.load('net.pth')
    modelT.train()
    
    criterion = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)

    # Maybe play around with lr
    optimizer = torch.optim.AdamW(modelT.parameters(), lr=0.0001)

    print('NYU base loading...')
    f = open('nyu_buffer.pickle', 'rb')
    nyu_base = list(pkl.load(f))
    nyu_base = nyu_base[:32]
    f.close()

    train_num = 0
    while True:
        if buffer.qsize() >= max_buffer_size:
        # Create a dataset from the buffer for training
            
            dataset = [buffer.get() for _ in range(max_buffer_size)]
            dataset.extend(nyu_base)
            dataset = get_loader(dataset, max_buffer_size)

            # Training step
            # for epoch in range(100):
            train_start = time.time()
            running_loss = 0
            for batch_idx, batch in enumerate(dataset):
                print(f'{batch_idx + 1} / {len(dataset)}')
                images = batch['image'].to(device)
                depth = batch['depth'].to(device)

                # print(f'Images: {images}')
                # print(f'Depths: {depth.min()}, {depth.max()}')

                optimizer.zero_grad()
                outputs = modelT(images)

                loss = criterion(outputs, depth)
                running_loss += loss.item()
                loss.backward()
                
                optimizer.step()

            train_end = time.time()

            # if train_num < 125:
            #     with open('train_time.csv', 'a+') as f:
            #         f.write(f"{train_end - train_start}\n")
            # else:
            #     print('Reached 125 train_num')
            #     exit(-1)

            print(f"Trained on a batch of {max_buffer_size} images. Loss: {running_loss / len(dataset)}:f over {train_end - train_start}s")

            train_num += 1

            # For unpruned PyTorch
            # modelT.save_weights("net_full.pth")
                
            # # For pruned PyTorch
            # torch.save(modelT, 'net.pth')
            
            if weight_buffer.empty():
                weight_buffer.put(modelT)
        else:
            time.sleep(1)  # Wait for the buffer to fill

# # Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Assuming 'Net' is your PyTorch model AND it is not pruned
# modelG = load_model('GuideDepth').to(device)
# modelG.reload_weights('GuideDepth/model/NYU_Half_GuideDepth.pth')
# modelG.train()  # Make sure the model is in training mode
# modelG.save_weights()

# from thop import profile

# modelG.eval()
# with torch.no_grad():
#     input_data = torch.randn(1, 3, 240, 320).cuda()
#     macs, params = profile(modelG, inputs=(input_data,))
#     print(macs)
#     print(params)

# exit(-1)

# Assuming you have a pruned PyTorch checkpoint
modelG = torch.load('prune/ignored_pruned_model_0.3.pt')
modelG.train()
torch.save(modelG, 'net.pth')

modelG = None
# # Criterion and optimizer

# The buffer to store data for training
buffer = queue.Queue()
weight_buffer = queue.Queue(maxsize=1)
max_buffer_size = 8  # Define the maximum size of the buffer

# Stop event to cleanly exit threads
stop_event = threading.Event()
def run_oit():
    # Starting the threads
    inference = threading.Thread(target=inference_thread)
    training = threading.Thread(target=training_thread)

    inference.start()
    training.start()

    # Running for a limited time or until a certain condition, then stopping
    # try:
    #     inference.join(timeout=10)  # Let's say we run this for 10 seconds for demonstration
    #     training.join(timeout=10)
    # except KeyboardInterrupt:
    #     pass
    #
    # stop_event.set()
    #
    # inference.join()
    # training.join()
    # return None

if __name__ == '__main__':
    run_oit()