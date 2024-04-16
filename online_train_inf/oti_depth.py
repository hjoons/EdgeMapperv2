import threading
import queue
import torch
import torchvision.transforms as transforms
import time
from GuideDepth.model.loader import load_model
import matplotlib.pyplot as plt
from pyk4a import PyK4A, Config, FPS, DepthMode, ColorResolution
import cv2
from dataloader import get_loader, get_dataset
import pickle as pkl
from loss import Depth_Loss
import numpy as np
from eval import compute_errors
import copy
import random
import pandas as pd
import torch.nn.utils.prune as prune
import onnxruntime as rt
import argparse
from datareader import quantize
parser = argparse.ArgumentParser(description='Run online training and inference')

parser.add_argument('--inf_type', type=str, default="pytorch", help="Run on either Pytorch or ONNX")

args = parser.parse_args()

RESOLUTION = (240,320)
BATCH_SIZE = 4
SAMPLE_PROB = 0.75
KEEP_PROB = 0.5
MAX_BUFFER_SIZE = 16

def inference_thread():
    resize_transform = transforms.Resize(RESOLUTION)
    fps = 30
    time_per_frame = 1 / fps  # 0.041666666666666664 = 41.6ms
    # Pre-trained weights
    model_path = "net.pth"

    if args.inf_type == 'pytorch':
        modelI = torch.load(model_path)
        modelI.eval()
    else:
        sess_options = rt.SessionOptions() # i don't think we need this ## I thiunk we do (Allen)
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = "temp.onnx"
        sess = rt.InferenceSession("temp.onnx", sess_options=sess_options, providers=["CUDAExecutionProvider"])
        input_name = sess.get_inputs()[0].name
    
    print(f"Starting capture...")
    config = Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_30
    )

    k4a = PyK4A(config)
    height = 480
    width = 640
    start_height = (720 - height) // 2
    start_width = (1280 - width) // 2
    end_height = start_height + height
    end_width = start_width + width


    keys = ['capture', 'color_train', 'depth_train', 'color_plot', 'depth_plot', 'sample', 'inference', 'inference_plot', 'plot']
    profile = {key: [] for key in keys}
    frame_num = 0
    
    # Open the device
    k4a.start()

    while True:
        capture_start = time.time()
        capture = k4a.get_capture()
        capture_end = time.time()

        profile['capture'].append(capture_end - capture_start)
        
        start_time_all = time.time()
        if stop_event.is_set():
            break
        

        if weight_buffer.full():
            print('Updating model...')
            if args.inf_type == 'pytorch':
                modelI = weight_buffer.get()
                modelI.eval()
            else: # ONNX
                message = weight_buffer.get()
                sess_options = rt.SessionOptions() # i don't think we need this ## I thiunk we do (Allen)
                sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.optimized_model_filepath = "quantized.onnx"
                sess = rt.InferenceSession("temp.onnx", sess_options=sess_options, providers=["CUDAExecutionProvider"])
                input_name = sess.get_inputs()[0].name
            print('Model updated!\n')

        
        if capture is not None:

            color_start = time.time()
            # Get the color image from the capture
            color_image = capture.color[start_height:end_height, start_width:end_width, 0:3]

            # resizing to (240,320) for downsampling
            color_image = torch.from_numpy(color_image).transpose(0,1).transpose(0,2)
            color_image = resize_transform(color_image)
            color_image = color_image.transpose(0,2).transpose(0,1).numpy()
            color_end = time.time()

            profile['color_train'].append(color_end - color_start)
            
            depth_start = time.time()
            # transformed_depth_image
            transformed_depth_image = capture.transformed_depth[start_height:end_height, start_width:end_width]

            # inpainting depth image
            # mask = (transformed_depth_image == 0).astype(np.uint8)
            # transformed_depth_image = cv2.inpaint(transformed_depth_image, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            depth_end = time.time()

            profile['depth_train'].append(depth_end - depth_start)
            
            depth_start = time.time()
            # resizing to (240,320) for downsampling
            transformed_depth_image_plt = torch.from_numpy(transformed_depth_image.astype(np.int32))
            transformed_depth_image_plt = resize_transform(transformed_depth_image_plt.unsqueeze(0))
            transformed_depth_image_plt = transformed_depth_image_plt.squeeze(0).numpy()

            valid_mask = (transformed_depth_image_plt != 0)
            valid_mask = np.expand_dims(valid_mask, axis=-1)
            transformed_depth_image_plt = transformed_depth_image_plt / 10000.0
            
            transformed_depth_image_plt = (transformed_depth_image_plt * 255).astype(np.uint8)        
            transformed_depth_image_plt = cv2.applyColorMap(transformed_depth_image_plt, cv2.COLORMAP_JET)
            transformed_depth_image_plt = transformed_depth_image_plt * valid_mask
            depth_end = time.time()

            profile['depth_plot'].append(depth_end - depth_start)

            # depth_np = capture.transformed_depth[start_height:end_height, start_width:end_width]
            depth_np = transformed_depth_image
            color_np = color_image
            
            color_start = time.time()
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)
            color_image_tensor = torch.from_numpy(color_image_rgb)
            color_image_tensor = color_image_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            color_image_tensor = color_image_tensor.float().div(255)
            color_end = time.time()

            profile['color_plot'].append(color_end - color_start)
            
            sample_start = time.time()
            
            # sampling for training
            if random.random() < SAMPLE_PROB:
                # Put the (image, label) tuple in the buffer
                if buffer.qsize() >= MAX_BUFFER_SIZE:
                    # Remove the oldest item to make space for the new one
                    buffer.get()  # This line removes the oldest entry from the queue

                buffer.put((color_np, depth_np))  # Move image back to CPU before storing in buffer
                sample_end = time.time()

                profile['sample'].append(sample_end - sample_start)
            if args.inf_type == 'pytorch':
                model_input = color_image_tensor.unsqueeze(0).to(device)
            else: # ONNX
                model_input = color_image_tensor.unsqueeze(0).to('cpu').numpy()
            with torch.no_grad():
                inf_start = time.time()
                if args.inf_type == 'pytorch':
                    pred = modelI(model_input)
                else: #ONNX
                    pred = sess.run(None, {input_name: model_input})[0]
                inf_end = time.time()

                profile['inference'].append(inf_end - inf_start)

                inf_start = time.time()

                if args.inf_type == 'pytorch':
                    pred = pred.detach().squeeze(0).squeeze(0).cpu().numpy()
                else: # ONNX 
                    pred = np.squeeze(pred)
                # pred_onnx = np.squeeze(pred_onnx, axis=0)
                # pred_onnx = np.squeeze(pred_onnx, axis=0)
                # zero_mask = pred == 0.0
                zero_mask = pred == 0.0
                
                # pred = np.clip(pred, 10.0/100.0, 10.0)
                pred = np.clip(pred, 10.0/100.0, 10.0) 
                # pred = 10.0 / pred
                pred = 10.0 / pred

                # pred[zero_mask] = 0.0
                pred[zero_mask] = 0.0

                # pred = ((pred / 10.0) * 255).astype(np.uint8)
                pred = ((pred / 10.0) * 255).astype(np.uint8)
                # pred = cv2.(pred, cv2.COLORMAP_JET)
                pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
                inf_end = time.time()
                profile['inference_plot'].append(inf_end - inf_start)

            plot_start = time.time()
            cv2.rectangle(color_image_rgb, (0,220), (75,240), color=(0,0,0), thickness=cv2.FILLED)
            cv2.putText(color_image_rgb, '{:.2f}fps'.format(1 / (time.time() - start_time_all)), (0,235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow("Real-Time Video", color_image_rgb)
            cv2.imshow("Depth Image", transformed_depth_image_plt)
            cv2.imshow("Prediction", pred)
            plot_end = time.time()

            frame_num += 1

            profile['plot'].append(plot_end - plot_start)
            if frame_num == 1000:
                for key in profile:
                    profile[key] += [None] * (1000 - len(profile[key]))

                df = pd.DataFrame.from_dict(profile)
                df.to_csv(f'profile__onnxqt_bs{BATCH_SIZE}.csv')
                print('\n\nFinished Profiling\n\n')

            time.sleep(max(0., time_per_frame - (time.time() - start_time_all)))

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    # Stop the cameras and close the device
    k4a.device_stop_cameras()
    k4a.device_close()

    # Close the OpenCV window
    cv2.destroyAllWindows()

def prune_model(model, pruning_ratio):
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
    


def training_thread():
    start_time = None
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
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)

    # Assuming that you are using a pruned PyTorch model
    modelT = torch.load('net.pth')
    
    criterion = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)

    optimizer = torch.optim.AdamW(modelT.parameters(), lr=0.001)

    # Base NYUv2 buffer
    f = open('nyu2_test1.pickle', 'rb')
    nyu_base = list(pkl.load(f))
    nyu_base = nyu_base[:16]
    f.close()

    train_num = 0
    while True:
        if buffer.qsize() >= MAX_BUFFER_SIZE:
        # Create a dataset from the buffer for training            
            inf_dataset = [buffer.get() for _ in range(MAX_BUFFER_SIZE)]

            # keep some older images, i.e. replay buffer
            for ind in inf_dataset:
                if random.random() < KEEP_PROB:
                    buffer.put((copy.deepcopy(ind[0]), copy.deepcopy(ind[1])))

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
            if (len(epochs) <= 0):
                epochs.append(0)
                start_time = time.time()
            else:
                epochs.append(time.time() - start_time)
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
        
            print('Trained on a batch of {:d} images. Loss: {:.5f} over {:.5f}s'.format(MAX_BUFFER_SIZE + len(nyu_base), running_loss / len(dataset), train_end - train_start))

            train_num += 1
            # print(f"Unstructured Pruning started...")
            # prune_model(modelT, 0.5)
            # print(f"Finished Unstructured Pruning")
            # for name, module in modelT.named_modules():
            #     if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            #         prune.l1_unstructured(module, name='weight', amount=0.8)
            #         prune.remove(module, 'weight')
            # Exchange models with inference thread
            if weight_buffer.empty():
                modelT.eval()
                if args.inf_type == 'pytorch':
                    weight_buffer.put(copy.deepcopy(modelT))
                else:
                    # quantize
                    oti_ds = get_dataset(inf_dataset,'train', RESOLUTION)
                    torch.onnx.export(modelT.to('cpu'), torch.randn(1,3,240,320), "temp.onnx", export_params=True, opset_version=17, do_constant_folding=True)
                    quantize('temp.onnx', 'quantized.onnx', oti_ds)
                    weight_buffer.put("ONNX model sent")
                modelT.to('cuda')

            
            
        else:
            time.sleep(1)  # Wait for the buffer to fill
# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have a pruned PyTorch checkpoint
modelG = torch.load('prune/ignored_pruned_model_0.3.pt')
modelG.train()
torch.save(modelG, 'net.pth')
modelG.eval()
if args.inf_type == 'onnx':
    torch.onnx.export(modelG.to('cpu'), torch.randn(1,3,240,320), "temp.onnx", export_params=True, opset_version=13, do_constant_folding=True)

modelG = None
# The buffer to store data for training
buffer = queue.Queue(maxsize=MAX_BUFFER_SIZE)
weight_buffer = queue.Queue(maxsize=1)

# Stop event to cleanly exit threads
stop_event = threading.Event()
def run_oit():
    # Starting the threads
    inference = threading.Thread(target=inference_thread)
    training = threading.Thread(target=training_thread)

    inference.start()
    training.start()

if __name__ == '__main__':
    run_oit()