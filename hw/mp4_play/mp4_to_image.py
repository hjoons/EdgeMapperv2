import cv2
import os

def mp4_to_images(video_path, output_folder):

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the desired frame rate for saving images
    desired_fps = 15

    # Calculate the frame interval to achieve the desired frame rate
    frame_interval = round(fps / desired_fps)

    # Variable to keep track of the frame number
    frame_number = 0

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if it's time to save the frame
        if frame_number % frame_interval == 0:
            # Save the frame as an image
            image_path = os.path.join(output_folder, f'frame_{frame_number:04d}.jpg')
            cv2.imwrite(image_path, frame)
        
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Video frames saved as images in '{output_folder}' at {desired_fps} fps.")
