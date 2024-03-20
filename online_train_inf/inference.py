from trt_inference import TRTModel
from pyk4a import PyK4A, Config, FPS, DepthMode, ColorResolution
import cv2
import time
import os

def run_inference(trt_model_path='../mbnv3_fp16.trt'):
    """
    This function performs inference using TensorRT and uses the model from trt_model_path.
    If the trt_model_path is updated, it is reloaded automatically.

    Args:
        trt_model_path: The path to the TensorRT model
    """
    # camera inference using TensorRT model
    model = TRTModel(trt_model_path)
    print('Model Loaded!')
    print(f"Starting capture...")

    config = Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_15
        # color_format=pyk4a.ImageFormat.COLOR_BGRA32,
    )
    k4a = PyK4A(config)
    k4a.start()
    fps = 24
    time_per_frame = 1 / fps  # 0.041666666666666664 = 41.6ms
    last_mod_time = os.path.getmtime(trt_model_path)
    while True:
        start_time_all = time.time()
        if os.path.getmtime(trt_model_path) != last_mod_time:
            model.reload_model(trt_model_path)
        # Get a capture
        capture = k4a.get_capture()
        if capture is not None:
            # Get the color image from the capture
            color_image = capture.color
            transformed_depth_image = capture.transformed_depth
            transformed_depth_image = cv2.normalize(transformed_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            transformed_depth_image = cv2.applyColorMap(transformed_depth_image, cv2.COLORMAP_JET)[120:600, 320:960, 0:1]

            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)[120:600, 320:960, 0:3]

            start_time = time.perf_counter()
            pred = model.inference(color_image_rgb)
            end_time = time.perf_counter()
            print(end_time - start_time)
            pred = 1000 / pred[0][0]
            pred = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

            cv2.imshow("Real-Time Video", color_image_rgb)
            cv2.imshow("Depth Image", transformed_depth_image)
            cv2.imshow("Prediction", pred)

            time.sleep(max(0., time_per_frame - (time.time() - start_time_all)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Stop the cameras and close the device
    k4a.device_stop_cameras()
    k4a.device_close()

    # Close the OpenCV window
    cv2.destroyAllWindows()