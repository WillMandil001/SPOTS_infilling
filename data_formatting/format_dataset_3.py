import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R

def create_rlds_dataset(folders, data_location, gelsight=False):
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
            if not gelsight: xela_sensor     = np.array(pd.read_csv(data_location + folder + '/xela_sensor1.csv', header=None))[1:]
            elif gelsight:   gelsight_sensor = np.array(np.load(data_location + folder + '/gelsight_images.npy'))
            # image_data  = np.array(np.load(data_location + folder + '/color_images.npy'))

            # for the original marked_object dataset
            # robot_state = np.array(pd.read_csv(data_location + folder + '/robot_states.csv', header=None))[1:]
            # xela_sensor = np.array(np.load(data_location + folder + '/tactile_states.npy'))
            image_data  = np.array(np.load(data_location + folder + '/color_images.npy'))
            robot_task_space = np.array(np.load(data_location + folder + '/robot_EE_states.npy'))
            # object_class = np.load(data_location + folder + '/classification_bit.npy')

        except:
            print("Error loading data from folder: ", folder)
            continue

        # robot_task_space = np.array([[state[-7], state[-6], state[-5]] + list(R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)) for state in robot_state]).astype(float)
        if not gelsight:
            tactile_data_split = [np.array(xela_sensor[:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]
            tactile_mean_start_values = [int(sum(tactile_data_split[feature][0]) / len(tactile_data_split[feature][0])) for feature in range(3)]
            tactile_offsets = [[tactile_mean_start_values[feature] - tactile_starting_value for tactile_starting_value in tactile_data_split[feature][0]] for feature in range(3)]
            tactile_data = [[tactile_data_split[feature][ts] + tactile_offsets[feature] for feature in range(3)] for ts in range(tactile_data_split[0].shape[0])]
        elif gelsight:
            gelsight_raw = []
            # for k in range(len(gelsight_sensor)):
            #     tmp = Image.fromarray(gelsight_sensor[k])
            #     tmp = tmp.resize((128, 128), Image.LANCZOS)
            #     tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
            #     tmp = tmp.reshape((128, 128, 3))
            #     gelsight_raw.append(tmp)
            for k in range(len(gelsight_sensor)):  #! we need to crop the gelsight images so that we dont add in the needless casing space around the image
                tmp = Image.fromarray(gelsight_sensor[k])
                width, height = tmp.size
                left   = 60
                top    = 90
                right  = width - 40
                bottom = height - 140
                tmp = tmp.crop((left, top, right, bottom))
                #save the image as a png
                tmp.save("gelsight_image.png")
                tmp = tmp.resize((128, 128), Image.LANCZOS)
                tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
                tmp = tmp.reshape((128, 128, 3))
                gelsight_raw.append(tmp)
            tactile_data = np.array(gelsight_raw)

        raw = []
        for k in range(len(image_data)):
            tmp = Image.fromarray(image_data[k])
            tmp = tmp.resize((128, 128), Image.LANCZOS)
            tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
            tmp = tmp.reshape((128, 128, 3))
            raw.append(tmp)

        # raw = []
        # for k in range(len(image_data)):
        #     tmp = Image.fromarray(image_data[k])
        #     width, height = tmp.size
        #     left   = width  / 4
        #     top    = height / 4
        #     right  = 3 * width  / 4
        #     bottom = 3 * height / 4
        #     tmp = tmp.crop((left, top, right, bottom))
        #     tmp = tmp.resize((128, 128), Image.LANCZOS)
        #     tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
        #     tmp = tmp.reshape((128, 128, 3))
        #     raw.append(tmp)

        image_data = np.array(raw)

        # tactile_data, robot_task_space, image_data

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
                'tactile': None,
                'action':  action.astype(np.float32),
                'language_instruction': "None",
                # 'object_class': int(object_class[0])
            }
            if not gelsight: step['tactile'] = np.array(tactile_data[i]).astype(np.float32)
            elif gelsight:   step['tactile'] = tactile_data[i]

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

# dataset_dirs = ["/media/wmandil/Data/Robotics/Data_sets/infilling_simple_005_6objs/train/",
#                 "/media/wmandil/Data/Robotics/Data_sets/infilling_simple_005_6objs/val/",
#                 "/media/wmandil/Data/Robotics/Data_sets/infilling_simple_005_6objs/test/"]

#! If using xela:
# dataset_dirs = ["/home/wmandil/robotics/datasets/infilling_simple_005_new_set/train/",
#                 "/home/wmandil/robotics/datasets/infilling_simple_005_new_set/val/",
#                 "/home/wmandil/robotics/datasets/infilling_simple_005_new_set/test/"]
# for dataset_dir in dataset_dirs:
#     create_rlds_dataset(folders=os.listdir(dataset_dir),  data_location=dataset_dir)

#! If using gelsight:
dataset_dirs = [#"/media/wmandil/Data/Robotics/Data_sets/infilling_simple_001_gelsight_unformatted/train/",
                "/media/wmandil/Data/Robotics/Data_sets/infilling_simple_001_gelsight_unformatted/val/",
                "/media/wmandil/Data/Robotics/Data_sets/infilling_simple_001_gelsight_unformatted/test/"]
for dataset_dir in dataset_dirs:
    create_rlds_dataset(folders=os.listdir(dataset_dir),  data_location=dataset_dir, gelsight=True)
