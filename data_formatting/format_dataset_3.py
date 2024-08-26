import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R

def create_rlds_dataset(folders, data_location):
    j = 0
    map = []
    for folder in folders:
        if "data_sample" not in folder:
            continue

        episode_list = []

        print("-------- ", folder, " --------")
        try:
            # for controlled test dataset
            # robot_state = np.array(pd.read_csv(data_location + folder + '/robot_state.csv', header=None))[1:]
            # xela_sensor = np.array(pd.read_csv(data_location + folder + '/xela_sensor1.csv', header=None))[1:]
            # image_data  = np.array(np.load(data_location + folder + '/color_images.npy'))

            # for the original marked_object dataset
            robot_state = np.array(pd.read_csv(data_location + folder + '/robot_states.csv', header=None))[1:]
            xela_sensor = np.array(np.load(data_location + folder + '/tactile_states.npy'))
            image_data  = np.array(np.load(data_location + folder + '/color_images.npy'))
        except:
            print("Error loading data from folder: ", folder)
            continue
        robot_task_space = np.array([[state[-7], state[-6], state[-5]] + list(R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)) for state in robot_state]).astype(float)
        tactile_data_split = [np.array(xela_sensor[:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]
        tactile_mean_start_values = [int(sum(tactile_data_split[feature][0]) / len(tactile_data_split[feature][0])) for feature in range(3)]
        tactile_offsets = [[tactile_mean_start_values[feature] - tactile_starting_value for tactile_starting_value in tactile_data_split[feature][0]] for feature in range(3)]
        tactile_data = [[tactile_data_split[feature][ts] + tactile_offsets[feature] for feature in range(3)] for ts in range(tactile_data_split[0].shape[0])]

        raw = []
        for k in range(len(image_data)):
            tmp = Image.fromarray(image_data[k])
            tmp = tmp.resize((64, 64), Image.LANCZOS)
            tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
            tmp = tmp.reshape((64, 64, 3))
            raw.append(tmp)

        image_data = np.array(raw)

        tactile_data, robot_task_space, image_data

        previous_state = robot_task_space[0]
        step_save_name_list = []
        # save episode as npy file
        dir = data_location + "formatted_dataset/episode_" + str(j) + "/"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
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
            step_save_dir = dir + "step_" + str(i) + ".npy"
            np.save(step_save_dir, np.array(step))
            step_save_name_list.append(step_save_dir)

        info = {
            "episode_save_dir": dir,
            "episode_length": len(episode_list),
            "episode_id": j,
            "step_save_name_list": step_save_name_list
        }

        map.append(info)

        j += 1

        np.save(data_location + "formatted_dataset/map.npy", map)

dataset_dirs = ["/media/wmandil/Data/Robotics/Data_sets/Dataset3_MarkedHeavyBox/train/",
                "/media/wmandil/Data/Robotics/Data_sets/Dataset3_MarkedHeavyBox/val/",
                "/media/wmandil/Data/Robotics/Data_sets/Dataset3_MarkedHeavyBox/test_examples/"]

for dataset_dir in dataset_dirs:
    create_rlds_dataset(folders=os.listdir(dataset_dir),  data_location=dataset_dir)