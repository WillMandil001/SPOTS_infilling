import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt

# def create_rlds_dataset(tactile_data, vision_data, data_location):
#     map = []
#     for episode_id, (tactile_episode, vision_episode) in enumerate(tqdm(zip(tactile_data, vision_data), total=len(tactile_data))):

#         # save episode as npy file
#         dir = data_location + "formatted_dataset/episode_" + str(episode_id) + "/"
#         if not os.path.exists(dir):
#             os.makedirs(dir, exist_ok=True)

#         step_save_name_list = []
#         episode_list = []
#         for i, (tactile_frame, vision_frame) in enumerate(zip(tactile_episode, vision_episode)):
#             tactile_frame = Image.open(tactile_frame)
#             vision_frame  = Image.open(vision_frame)

#             tactile_frame = tactile_frame.resize((256, 256), Image.LANCZOS)
#             vision_frame = vision_frame.resize((64, 64), Image.LANCZOS)

#             tactile_frame = np.frombuffer(tactile_frame.tobytes(), dtype=np.uint8)
#             vision_frame = np.frombuffer(vision_frame.tobytes(), dtype=np.uint8)

#             tactile_frame = tactile_frame.reshape((256, 256, 3))
#             vision_frame = vision_frame.reshape((64, 64, 3))

#             step = {
#                 'image':   vision_frame,
#                 'tactile': tactile_frame,
#             }

#             episode_list.append(step)
#             step_save_dir = dir + "step_" + str(i) + ".npy"
#             np.save(step_save_dir, np.array(step))
#             step_save_name_list.append(step_save_dir)

#         info = {"episode_save_dir": dir, 
#                 "episode_length": len(episode_list), 
#                 "episode_id": episode_id, 
#                 "step_save_name_list": step_save_name_list}

#         map.append(info)

#         # dont do this every time, is just in case the program crashes
#         if episode_id % 100 == 0:
#             np.save(data_location + "formatted_dataset/map.npy", map)

#     np.save(data_location + "formatted_dataset/map.npy", map)


def create_rlds_dataset(tactile_data, vision_data, data_location):
    map = []
    plt.ion()
    for episode_id, (tactile_episode, vision_episode) in enumerate(tqdm(zip(tactile_data, vision_data), total=len(tactile_data))):

        # save episode as npy file
        dir = data_location + "formatted_dataset/episode_" + str(episode_id) + "/"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        step_save_name_list = []
        episode_list = []

        # we dont want to all the time frames from the episode. (its too ling)
        # find the frames where the tactile data actually starts to be collected, then only store the next 50 frames, not all frames from the episode
        tactile_set = []
        for i, tactile_frame in enumerate(tactile_episode):
            tactile_set.append(np.array(Image.open(tactile_frame).resize((256, 256), Image.LANCZOS).convert('L')))

        change_metrics = []

        # work out the mean of the first 5 frames
        reference_frames = []
        for frame in tactile_set[:20]:
            frame_array = np.array(frame, dtype=np.float32)
            reference_frames.append(frame_array)
        reference_array = np.mean(reference_frames, axis=0)

        for idx, frame in enumerate(tactile_set):        # Compute change metrics for each frame
            diff = np.abs(frame - reference_array)
            change_metric = np.sum(diff)
            change_metrics.append(change_metric)

        # Determine a threshold based on the baseline (first 5 frames)
        baseline_metrics = change_metrics[:20]
        mean_metric = np.mean(baseline_metrics)
        std_metric = np.std(baseline_metrics)
        threshold = (mean_metric + 2 * std_metric)*2

        # Identify the start frame
        start_frame = next((idx for idx, metric in enumerate(change_metrics) if metric > threshold), None)

        # Identify the end frame
        end_frame = None
        if start_frame is not None:
            for idx in range(start_frame + 1, len(change_metrics)):
                if change_metrics[idx] < threshold:
                    end_frame = idx
                    break

        print(f"Tactile interaction starts at frame index: {start_frame}")
        print(f"Tactile interaction ends at frame index: {end_frame}")

        if start_frame is None and end_frame is None:
            print("No tactile interaction detected in episode")
            continue
        if start_frame is None: start_frame = 0
        if end_frame is None:   end_frame = len(tactile_set) - 1

        # add 15 to the end frame to get the next 15 frames
        end_frame = end_frame + 15
        if end_frame > len(tactile_set): end_frame = len(tactile_set)

        # minus 15 from the start frame to get the previous 15 frames
        start_frame = start_frame - 15
        if start_frame < 0: start_frame = 0

        # Plot the change metrics to visualize changes over time
        plt.figure(figsize=(10, 5))
        plt.plot(change_metrics, label='Change Metric')
        # add vertical lines to indicate the start and end of the tactile interaction
        if start_frame is not None:
            plt.axvline(x=start_frame, color='r', linestyle='--', label='Start of Tactile Interaction')
        if end_frame is not None:
            plt.axvline(x=end_frame, color='g', linestyle='--', label='End of Tactile Interaction')
        plt.xlabel('Frame Index')
        plt.ylabel('Sum of Absolute Differences')
        plt.title('Change Metric Over Time')
        plt.legend()
        plt.draw()
        plt.pause(0.001)

        # Save the frames where the tactile interaction starts
        for i, (tactile_frame, vision_frame) in enumerate(zip(tactile_episode[start_frame:end_frame], vision_episode[start_frame:end_frame])):
            tactile_frame = Image.open(tactile_frame)
            vision_frame  = Image.open(vision_frame)

            tactile_frame = tactile_frame.resize((64, 64), Image.LANCZOS)
            vision_frame = vision_frame.resize((64, 64), Image.LANCZOS)

            tactile_frame = np.frombuffer(tactile_frame.tobytes(), dtype=np.uint8)
            vision_frame = np.frombuffer(vision_frame.tobytes(), dtype=np.uint8)

            tactile_frame = tactile_frame.reshape((64, 64, 3))
            vision_frame = vision_frame.reshape((64, 64, 3))

            step = {
                'image':   vision_frame,
                'tactile': tactile_frame,
            }

            episode_list.append(step)
            step_save_dir = dir + "step_" + str(i) + ".npy"
            np.save(step_save_dir, np.array(step))
            step_save_name_list.append(step_save_dir)

        info = {"episode_save_dir": dir, 
                "episode_length": len(episode_list), 
                "episode_id": episode_id, 
                "step_save_name_list": step_save_name_list}

        map.append(info)

        # the plt show will block the program, so we need to close it
        plt.close()

        # dont do this every time, is just in case the program crashes
        if episode_id % 100 == 0:
            np.save(data_location + "formatted_dataset/map.npy", map)

    np.save(data_location + "formatted_dataset/map.npy", map)

# set the location of the dataset
dataset_location         = "/media/wmandil/Data/Robotics/Data_sets/infilling_simple_001_gelsight"

dataset_location         = "/media/wmandil/Data/Robotics/Data_sets/VisGel/data/data_unseen/images/"
dataset_location_tactile = dataset_location + "touch/"
dataset_location_vision  = dataset_location + "vision/"

# load all the folders in the dataset touch
folders_tactile = os.listdir(dataset_location_tactile)
folders_vision = os.listdir(dataset_location_vision)

# add the episodes from each folder to the list of episodes
episodes_tactile = []
for folder in folders_tactile:
    for episode in os.listdir(dataset_location_tactile + folder):
        episodes_tactile.append(dataset_location_tactile + folder + "/" + episode)
episodes_vision  = []
for folder in folders_vision:
    for episode in os.listdir(dataset_location_vision + folder):
        episodes_vision.append(dataset_location_vision + folder + "/" + episode)

# get the frames in each episode
episodes_tactile_set = []
for episode in episodes_tactile:
    files = os.listdir(episode)
    episodes_tactile_set.append(sorted([episode + "/" + f for f in files if f.startswith("frame") and f.endswith(".jpg")], key=lambda x: int(x.split('/')[-1].replace("frame", "").split('.')[0])))
episodes_vision_set  = []
for episode in episodes_vision:
    files = os.listdir(episode) 
    episodes_vision_set.append(sorted([episode + "/" + f for f in files if f.startswith("frame") and f.endswith(".jpg")], key=lambda x: int(x.split('/')[-1].replace("frame", "").split('.')[0])))

###### RUN THE RLDS CREATOR ######
create_rlds_dataset(tactile_data=episodes_tactile_set, vision_data=episodes_vision_set, data_location=dataset_location)
