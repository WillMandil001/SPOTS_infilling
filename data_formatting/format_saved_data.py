import os
import cv2

def rescale_images_in_folder(base_folder, target_size=(64, 64)):
    """
    Rescale all images in the folder structure to the target size.
    
    Parameters:
    - base_folder (str): The root folder containing 'touch' directories.
    - target_size (tuple): The target size for resizing (width, height).
    """
    # Loop through each 'touch' folder
    for touch_folder in os.listdir(base_folder):
        touch_path = os.path.join(base_folder, touch_folder)
        
        if os.path.isdir(touch_path):
            print(f"Processing {touch_folder}...")
            # Loop through each 'rec_XXXX' folder
            for rec_folder in os.listdir(touch_path):
                rec_path = os.path.join(touch_path, rec_folder)
                
                if os.path.isdir(rec_path):
                    print(f"  Processing {rec_folder}...")
                    # Loop through each frame image
                    for frame_file in os.listdir(rec_path):
                        if frame_file.endswith('.jpg'):
                            frame_path = os.path.join(rec_path, frame_file)

                            # Read the image
                            img = cv2.imread(frame_path)
                            if img is not None:
                                # Resize the image
                                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                                # Overwrite the original image with the resized version
                                cv2.imwrite(frame_path, resized_img)
                                print(f" {touch_folder}  {rec_folder}  {frame_file}")
                            else:
                                print(f"    Failed to read {frame_file}, skipping.")
    print("Processing complete.")

# Path to the root folder (update this to your folder path)
base_folder = "/media/wmandil/Data/Robotics/Data_sets/VisGel/data/data_seen/images/touch/"

# Run the rescaling function
rescale_images_in_folder(base_folder)
