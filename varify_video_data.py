# load the numpy dataset and save the image sequence as a video

import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

image_width = 256
image_height = 256

def create_image(tactile):
    # convert tactile data into an image:
    image = np.zeros((4, 4, 3), np.float32)
    index = 0
    for x in range(4):
        for y in range(4):
            image[x][y] = [tactile[0][index],
                           tactile[1][index],
                           tactile[2][index]]
            index += 1
    reshaped_image = cv2.resize(image.astype(np.float32), dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    # convert to uint8 and scale to 0-255
    reshaped_image = (reshaped_image * 255).astype(np.uint8)

    # split each channel into its own image
    x = np.zeros((image_width, image_height, 3), np.uint8)
    y = np.zeros((image_width, image_height, 3), np.uint8)
    z = np.zeros((image_width, image_height, 3), np.uint8)

    x[:, :, 0] = reshaped_image[:, :, 0]
    y[:, :, 1] = reshaped_image[:, :, 1]
    z[:, :, 2] = reshaped_image[:, :, 2]

    return reshaped_image, x, y, z

def create_location_plot(robot_states, time_step, max_length, size=(image_width, image_height)):
    if len(robot_states) == 0:
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Get the min and max values for the axes
    x_values = [state[0] for state in robot_states]
    y_values = [state[1] for state in robot_states]
    epsilon = 1e-5  # Small value to avoid singular transformations

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    if x_min == x_max:
        x_min -= epsilon
        x_max += epsilon
    if y_min == y_max:
        y_min -= epsilon
        y_max += epsilon

    fig, ax = plt.subplots(figsize=(size[0] / 10, size[1] / 10), dpi=10)
    ax.plot(y_values, x_values, 'bo-')

    # Set limits based on overall trajectory length
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect('equal', 'box')
    plt.axis('off')

    fig.canvas.draw()
    plot_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plot_image = cv2.resize(plot_image, dsize=(size[0], size[1]), interpolation=cv2.INTER_CUBIC)
    plt.close(fig)
    return plot_image

map_file = "/home/wmandil/robotics/datasets/robot_pushing/train/formatted_dataset/map.npy"
map_data = np.load(map_file, allow_pickle=True)
to_replace   = "/media/wmandil/Data/Robotics/Data_sets/single_object_velocity_controlled_dataset/single_object_velocity_controlled_dataset/"
replace_with = "/home/wmandil/robotics/datasets/robot_pushing/"


for episode in map_data[1:]:
    episode_length = episode['episode_length']
    video_frames = []
    tactile_frames = []
    tactile_frames_x = []
    tactile_frames_y = []
    tactile_frames_z = []
    robot_states = []
    location_plots = []

    for step_num, save_name in enumerate(episode['step_save_name_list'][::2]):
        save_name = save_name.replace(to_replace, replace_with)            # overwrite location if it has changed:
        step_data = np.load(save_name, allow_pickle=True)
        image_data   = step_data[()]['image'].transpose(2, 0, 1)
        tactile_data = step_data[()]['tactile']
        robot_state  = step_data[()]["state"]

        video_frames.append(image_data)

        tactile_frame, tactile_frame_x, tactile_frame_y, tactile_frame_z = create_image(tactile_data)
        tactile_frames.append(tactile_frame)
        tactile_frames_x.append(tactile_frame_x)
        tactile_frames_y.append(tactile_frame_y)
        tactile_frames_z.append(tactile_frame_z)

        robot_states.append(robot_state[:2])

        print("Step: ", step_num, " of ", episode_length)

    max_length = len(robot_states)
    for i in range(len(robot_states)):
        location_plots.append(create_location_plot(robot_states[:i + 1], i + 1, max_length, size=(image_width, image_height)))

    # reshape the data
    video_frames = np.array(video_frames)
    video_frames = video_frames.transpose(0, 2, 3, 1)

    # flip rgb to bgr
    video_frames = video_frames[..., ::-1]

    # resize the images
    video_frames = [cv2.resize(frame, (image_width, image_height)) for frame in video_frames]

    # now we combine the tactile and video data into one long image that is 64 , 2*64, 3
    combined_frames = []
    for i in range(len(video_frames)):
        combined_frames.append(np.concatenate((video_frames[i], tactile_frames_x[i], tactile_frames_y[i], tactile_frames_z[i], location_plots[i]), axis=1))
    combined_frames = np.array(combined_frames)

    # remove half the fames, ever other frame
    video_frames = video_frames[::2]
    combined_frames = combined_frames[::2]

    # Save the collected frames as a GIF
    import imageio
    # imageio.mimsave("name_test4" + ".gif", video_frames, duration=0.01)
    # and make the gif repeat
    imageio.mimsave("name_test5" + ".gif", combined_frames, duration=0.01, loop=0)

    exit()

