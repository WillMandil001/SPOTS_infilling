import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R

def create_rlds_dataset(folders, data_location):
    j = 0
    for folder in folders:
        if "data_sample" not in folder:
            continue

        episode_list = []

        print("-------- ", folder, " --------")
        try:
            robot_state = np.array(pd.read_csv(data_location + folder + '/robot_state.csv', header=None))[1:]
            xela_sensor = np.array(pd.read_csv(data_location + folder + '/xela_sensor1.csv', header=None))[1:]
            image_data  = np.array(np.load(data_location + folder + '/color_images.npy'))
        except:
            print("Error loading data from folder: ", folder)
            continue
        # convert orientation to euler, and remove column labels:
        robot_task_space = np.array([[state[-7], state[-6], state[-5]] + list(R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)) for state in robot_state]).astype(float)

        # split tactile sensor into the three forces | find start value average for each force | find offsets for each taxel | take away start value average from the tactile data:
        tactile_data_split = [np.array(xela_sensor[:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]
        tactile_mean_start_values = [int(sum(tactile_data_split[feature][0]) / len(tactile_data_split[feature][0])) for feature in range(3)]
        tactile_offsets = [[tactile_mean_start_values[feature] - tactile_starting_value for tactile_starting_value in tactile_data_split[feature][0]] for feature in range(3)]
        tactile_data = [[tactile_data_split[feature][ts] + tactile_offsets[feature] for feature in range(3)] for ts in range(tactile_data_split[0].shape[0])]

        # Resize the image using PIL antialiasing method (Copied from CDNA data formatting)
        raw = []
        for k in range(len(image_data)):
            tmp = Image.fromarray(image_data[k])
            tmp = tmp.resize((64, 64), Image.LANCZOS)
            tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
            tmp = tmp.reshape((64, 64, 3))
            # tmp = tmp.astype(np.float32) / 255.0
            raw.append(tmp)
        image_data = np.array(raw)

        tactile_data, robot_task_space, image_data

        # need to save the the episode as a npy array with a dict defining each array: ['image', 'depth_image', 'state', 'action', 'language_instruction']
        previous_state = robot_task_space[0]
        for i in range(len(tactile_data)):
            state = robot_task_space[i]
            action = state - previous_state
            previous_state = state

            step = {
                'image':   image_data[i],
                'state':   robot_task_space[i].astype(np.float32),
                'tactile': np.array(tactile_data[i]).astype(np.float32),
                'action':  action.astype(np.float32),
                'language_instruction': "None"
            }
            episode_list.append(step)

        # save episode as npy file
        if not os.path.exists(os.path.join(data_location, "formatted_dataset")):
            os.makedirs(os.path.join(data_location, "formatted_dataset"))
        print("saving the episode as a npy file... episode_{}.npy".format(j))
        np.save(data_location + "formatted_dataset/episode_" + str(j) + ".npy", episode_list)
        j += 1

dataset_dirs = ["/media/wmandil/Data/Robotics/Data_sets/single_object_velocity_controlled_dataset/single_object_velocity_controlled_dataset/train/",
                "/media/wmandil/Data/Robotics/Data_sets/single_object_velocity_controlled_dataset/single_object_velocity_controlled_dataset/test/"]

for dataset_dir in dataset_dirs:
    create_rlds_dataset(folders=os.listdir(dataset_dir),  data_location=dataset_dir)