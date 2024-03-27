import threading
import queue
import torch
import torchvision.transforms as transforms
import time
from GuideDepth.model.loader import load_model
import matplotlib.pyplot as plt
from pyk4a import PyK4A, Config, FPS, DepthMode, ColorResolution
import cv2
from dataloader import get_loader
import pickle as pkl
from loss import Depth_Loss
import numpy as np
from eval import compute_errors
import copy
import random

RESOLUTION = (240,320)
BATCH_SIZE = 8
SAMPLE_PROB = 0.5
KEEP_PROB = 0.5

def inference_thread():
    resize_transform = transforms.Resize(RESOLUTION)
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
    height = 480
    width = 640
    start_height = (720 - height) // 2
    start_width = (1280 - width) // 2
    end_height = start_height + height
    end_width = start_width + width

    # Open the device
    k4a.start()

    # times = []

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
            print('Model updated!\n')

        if capture is not None:
            # Get the color image from the capture
            color_image = capture.color[start_height:end_height, start_width:end_width, 0:3]

            # resizing to (240,320) for downsampling
            color_image = torch.from_numpy(color_image).transpose(0,1).transpose(0,2)
            color_image = resize_transform(color_image)
            color_image = color_image.transpose(0,2).transpose(0,1).numpy()

            # transformed_depth_image
            transformed_depth_image = capture.transformed_depth[start_height:end_height, start_width:end_width]

            # inpainting depth image
            mask = (transformed_depth_image == 0).astype(np.uint8)
            transformed_depth_image = cv2.inpaint(transformed_depth_image, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

            # resizing to (240,320) for downsampling
            transformed_depth_image = torch.from_numpy(transformed_depth_image.astype(np.int32))
            transformed_depth_image = resize_transform(transformed_depth_image.unsqueeze(0))
            transformed_depth_image_np = transformed_depth_image.squeeze(0).numpy()

            # valid_mask = (transformed_depth_image_np != 0)
            # valid_mask = np.expand_dims(valid_mask, axis=-1)
            transformed_depth_image = transformed_depth_image_np / 10000.0
            
            transformed_depth_image_int8 = (transformed_depth_image * 255).astype(np.uint8)        
            transformed_depth_image_int8 = cv2.applyColorMap(transformed_depth_image_int8, cv2.COLORMAP_JET)
            # transformed_depth_image_int8 = transformed_depth_image_int8 * valid_mask


            # depth_np = capture.transformed_depth[start_height:end_height, start_width:end_width]
            depth_np = transformed_depth_image_np
            color_np = color_image
                        
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)
            color_image_tensor = torch.from_numpy(color_image_rgb)
            color_image_tensor = color_image_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            color_image_tensor = color_image_tensor.float().div(255)

            # sampling for training
            if random.random() < SAMPLE_PROB:
                # Put the (image, label) tuple in the buffer
                if buffer.qsize() >= max_buffer_size:
                    # Remove the oldest item to make space for the new one
                    buffer.get()  # This line removes the oldest entry from the queue

                buffer.put((color_np, depth_np))  # Move image back to CPU before storing in buffer

            # print("Added a new entry to the buffer.")

            model_input = color_image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                # pred_start = time.time()
                pred = modelI(model_input)
                # pred_end = time.time()

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
    epochs = []
    abs_rels = []
    sq_rels = []
    rmses = []
    rmse_logs = []
    d1s = []
    d2s = []
    d3s = []

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    fig.suptitle("Eval Metrics")
    absr, = ax1.plot([],[], color="blue", label="Abs Rel")
    sqr, = ax1.plot([],[], color="red", label="Sq Rel")
    rms, = ax1.plot([],[], color="orange", label="RMSE")
    rmsel, = ax1.plot([],[], color="green", label="RMSE log")
    ax1.legend(loc='upper right', bbox_to_anchor=(1,1))
    ax1.grid(True)

    d_1, = ax2.plot([],[], color="purple", label=r'$\delta$1')
    d_2, = ax2.plot([],[], color="pink", label=r'$\delta$2')
    d_3, = ax2.plot([],[], color="brown", label=r'$\delta$3')
    ax2.legend(loc='lower right', bbox_to_anchor=(1,0))
    ax2.set_xlabel("Epochs")
    ax2.grid(True)

    
    # # Assuming that you are not using a pruned PyTorch model    
    # modelT = load_model('GuideDepth').to(device)
    # modelT.train()
    # modelT.reload_weights()

    # Assuming that you are using a pruned PyTorch model
    modelT = torch.load('net.pth')
    
    criterion = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)

    # Maybe play around with lr
    optimizer = torch.optim.AdamW(modelT.parameters(), lr=0.001)

    # print('NYU base loading...')
    f = open('nyu2_base.pickle', 'rb')
    nyu_base = list(pkl.load(f))
    nyu_base = nyu_base[:16]
    f.close()

    train_num = 0
    while True:
        if buffer.qsize() >= max_buffer_size:
        # Create a dataset from the buffer for training            

            inf_dataset = [buffer.get() for _ in range(max_buffer_size)]

            # keep some older images, i.e. replay buffer
            for ind in inf_dataset:
                if random.random() < KEEP_PROB:
                    buffer.put((copy.deepcopy(ind[0]), copy.deepcopy(ind[1])))
            
            
            # with open('dataloader.pkl', 'wb') as f:
            #     pkl.dump(dataset, f)

            print('Evaluating model...')
            modelT.eval()
            dataset = get_loader(copy.deepcopy(inf_dataset), BATCH_SIZE, RESOLUTION, 'eval')
            # Eval
            errors = []
            running_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataset):
                    print(f'{batch_idx + 1} / {len(dataset)}')
                    images = batch['image'].to(device)
                    depth = batch['depth'].to(device)

                    outputs = modelT(images)

                    outputs = 10.0 / outputs
                    outputs = torch.clamp(outputs, 10.0 / 100.0, 10.0)
                    errors.append(compute_errors(depth, outputs))

            error_tensors = [torch.tensor(e).to('cuda') for e in errors]
            error_stack = torch.stack(error_tensors, dim=0)
            mean_errors = error_stack.mean(0).cpu().numpy()
            
            abs_rel = mean_errors[0]
            sq_rel = mean_errors[1]
            rmse = mean_errors[2]
            rmse_log = mean_errors[3]
            d1 = mean_errors[4]
            d2 = mean_errors[5]
            d3 = mean_errors[6]

            epochs.append(train_num)
            abs_rels.append(abs_rel)
            sq_rels.append(sq_rel)
            rmses.append(rmse)
            rmse_logs.append(rmse_log)
            d1s.append(d1)
            d2s.append(d2)
            d3s.append(d3)
            # print(f'abs_rel: {abs_rel}\nsq_rel: {sq_rel}\nrmse: {rmse}\nrmse_log: {rmse_log}\nd1: {d1}\nd2: {d2}\n d3: {d3}\n')

            absr.set_data(epochs, abs_rels)
            sqr.set_data(epochs, sq_rels)
            rms.set_data(epochs, rmses)
            rmsel.set_data(epochs, rmse_logs)
            d_1.set_data(epochs, d1s)
            d_2.set_data(epochs, d2s)
            d_3.set_data(epochs, d3s)

            ax1.relim()
            ax1.autoscale_view()
            ax2.set_ylim(0,1)
            ax2.autoscale_view()
            plt.draw()
            plt.pause(0.1)
            print('Model Evaluated!\n')
            
            modelT.train()
            inf_dataset.extend(nyu_base)
            dataset = get_loader(inf_dataset, BATCH_SIZE, RESOLUTION, 'train')
            print('Training model...')
            # Training step            
            train_start = time.time()
            for batch_idx, batch in enumerate(dataset):
                print(f'{batch_idx + 1} / {len(dataset)}')
                images = batch['image'].to(device)
                depth = batch['depth'].to(device)

                optimizer.zero_grad()
                outputs = modelT(images)

                loss = criterion(outputs, depth)
                running_loss += loss.item()
                loss.backward()
                
                optimizer.step()

            train_end = time.time()
        
            print('Trained on a batch of {:d} images. Loss: {:.5f} over {:.5f}s'.format(max_buffer_size + len(nyu_base), running_loss / len(dataset), train_end - train_start))

            # if train_num < 125:
            #     with open('train_time.csv', 'a+') as f:
            #         f.write(f"{train_end - train_start}\n")
            # else:
            #     print('Reached 125 train_num')
            #     exit(-1)
            # For unpruned PyTorch
            # modelT.save_weights("net_full.pth")
                
            # # For pruned PyTorch
            # torch.save(modelT, 'net.pth')
            train_num += 1
                
            if weight_buffer.empty():
                weight_buffer.put(copy.deepcopy(modelT))

            
            
        else:
            time.sleep(1)  # Wait for the buffer to fill

# # Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Assuming 'Net' is your PyTorch model AND it is not pruned
# modelG = load_model('GuideDepth').to(device)
# modelG.reload_weights('GuideDepth/model/NYU_Half_GuideDepth.pth')
# modelG.train()  # Make sure the model is in training mode
# modelG.save_weights()

# Assuming you have a pruned PyTorch checkpoint
modelG = torch.load('prune/ignored_pruned_model_0.3.pt')
modelG.train()
torch.save(modelG, 'net.pth')

modelG = None

# The buffer to store data for training
max_buffer_size = 16  # Define the maximum size of the buffer
buffer = queue.Queue(maxsize=max_buffer_size)
weight_buffer = queue.Queue(maxsize=1)

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