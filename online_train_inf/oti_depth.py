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
import matplotlib.pyplot as plt
import pyk4a
from pyk4a import PyK4A, Config, FPS, DepthMode, ColorResolution
import h5py

def inference_thread():
    fps = 24
    time_per_frame = 1 / fps  # 0.041666666666666664 = 41.6ms
    # CIFAR10 class labels
    model_path = "net.pth"

    modelI = MobileNetSkipConcat().to(device)
    modelI.eval()
    modelI.reload_weights()
    last_mod_time = os.path.getmtime(model_path)

    print(f"Starting capture...")
    config = Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_15
        #color_format=pyk4a.ImageFormat.COLOR_BGRA32,
    )

    k4a = PyK4A(config)
    height = 480
    width = 640
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
        if os.path.getmtime(model_path) != last_mod_time:
            modelI.reload_weights()
            modelI.eval()
            last_mod_time = os.path.getmtime(model_path)

        if capture is not None:
            # Get the color image from the capture
            color_image = capture.color
            depth_image = capture.depth
            transformed_depth_image = capture.transformed_depth
            transformed_depth_image = cv2.normalize(transformed_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            transformed_depth_image = cv2.applyColorMap(transformed_depth_image, cv2.COLORMAP_JET)
            transformed_depth_image = transformed_depth_image[start_height:end_height, start_width:end_width]
            
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)[start_height:end_height, start_width:end_width, 0:3]
            color_image_tensor = torch.from_numpy(color_image_rgb)
            color_image_tensor = color_image_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            color_image_tensor = color_image_tensor.float().div(255)

            # Put the (image, label) tuple in the buffer
            if buffer.qsize() >= max_buffer_size:
                # Remove the oldest item to make space for the new one
                buffer.get()  # This line removes the oldest entry from the queue
                print("Removed the oldest entry from the buffer.")
            buffer.put((color_image_tensor.cpu(), transformed_depth_image.cpu()))  # Move image back to CPU before storing in buffer
            print("Added a new entry to the buffer.")

            model_input = color_image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = modelI(model_input)
                pred = pred.detach().squeeze(0).squeeze(0).cpu().numpy()
                pred = 1000 / pred
                pred = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            
            cv2.imshow("Real-Time Video", color_image_rgb)
            cv2.imshow("Depth Image", transformed_depth_image)
            cv2.imshow("Prediction", pred)
            time.sleep(max(0., time_per_frame - (time.time() - start_time_all)))
            
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    # Stop the cameras and close the device
    k4a.device_stop_cameras()
    k4a.device_close()

    # Close the OpenCV window
    cv2.destroyAllWindows()


def training_thread():
    modelT = MobileNetSkipConcat().to(device)
    modelT.train()
    modelT.reload_weights()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(modelT.parameters(), lr=0.001, momentum=0.9)
    # while not stop_event.is_set() or not buffer.empty():
    while True:
        # if buffer.qsize() >= max_buffer_size:
        #     # Create a dataset from the buffer for training
        #     dataset = [buffer.get() for _ in range(max_buffer_size)]
        #     images, labels = zip(*dataset)
        #     images = torch.stack(images).to(device)
        #     labels = torch.tensor(labels).to(device)

        #     # Training step
        #     optimizer.zero_grad()
        #     outputs = modelT(images)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()

        #     print(f"Trained on a batch of {len(images)} images. Loss: {loss.item()}")
        #     modelT.save_weights("net.pth")
        # else:
        time.sleep(1)  # Wait for the buffer to fill

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR10 Data loading and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Assuming 'Net' is your PyTorch model
modelG = MobileNetSkipConcat().to(device)
modelG.train()  # Make sure the model is in training mode
modelG.save_weights()
modelG = None
# Criterion and optimizer


# The buffer to store data for training
buffer = queue.Queue()
max_buffer_size = 128  # Define the maximum size of the buffer

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