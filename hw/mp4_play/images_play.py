import cv2
import os
import time

# Path to the folder containing images
def images_play(path):
    image_folder = path

    # Get the list of image files
    image_files = sorted(os.listdir(image_folder))

    # Set the desired frame rate for display
    desired_fps = 45 # i had to triple is from 15 fps to get it to play in real time

    # Calculate the time interval between each frame
    frame_interval = 1 / desired_fps

    # Loop through the images and display them
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            cv2.imshow('Image', frame)
            # Wait for the calculated time interval
            time.sleep(frame_interval)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
