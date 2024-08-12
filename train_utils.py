

"""
Contains trajectory transforms used in the octo data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory
length).
"""

"""
Contains observation-level transforms used in the octo data pipeline. These transforms operate on the
"observation" dictionary, and are applied at a per-frame level.
"""

import flax
import time
import wandb
import torch
import os
import numpy as np
import cv2
# from absl import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable

import torch.nn as nn
import matplotlib.pyplot as plt

from flax.traverse_util import flatten_dict
from matplotlib.gridspec import GridSpec


###########################
#
# loss and cost functions
#
###########################

def get_object_angle(image_sequence):
    '''
    This function takes in a sequence of images and returns the angle of the object in the image
    - The angle is with respect to the horizontal axis in the image
    - we will use cv2 functions to iscolate the object and then calculate the principle component of the object to find the dominant angle
    - we will return the angle in degrees
    '''

    object_angle_list = []

    image = image_sequence[0].permute(1, 2, 0).cpu().numpy()
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(imageGray, 55, 255, cv2.THRESH_BINARY)
    des = cv2.bitwise_not(thresh)


    return object_angle_list


###########################
#
# Vizualisation functions
#
###########################

def viz_image_figure(ground_truth, predicted_frames, input_frames, config, step, step_name):
    if len(predicted_frames.shape) == 5: predicted_frames = predicted_frames.squeeze()

    if config.prediction_horizon > 20:
        sample_rate = config.prediction_horizon // 20
    else:
        sample_rate = 1

    if config.infill_patches:
        fig = plt.figure(figsize=(config.prediction_horizon * 0.25, 5))
        gs = GridSpec(3, 20, figure=fig)
    else:
        fig = plt.figure(figsize=(config.prediction_horizon * 0.25, 5))
        gs = GridSpec(2, 20, figure=fig)

    index = 0
    for j in range(sample_rate - 1, config.prediction_horizon, sample_rate):
        if config.infill_patches: ax = fig.add_subplot(gs[1, index])
        else:                     ax = fig.add_subplot(gs[0, index])
        ax.imshow(ground_truth[0][j].permute(1, 2, 0).cpu().numpy()[..., ::-1])
        ax.axis('off')
        ax.set_title(f"GT {j + 1}")
        if config.infill_patches: ax = fig.add_subplot(gs[2, index])
        else:                     ax = fig.add_subplot(gs[1, index])
        ax.imshow(predicted_frames[j].permute(1, 2, 0).cpu().numpy()[..., ::-1])
        ax.axis('off')
        ax.set_title(f"Pred {j + 1}")
        index += 1

    if config.infill_patches:
        for j in range(input_frames.shape[1]):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(input_frames[0][j].permute(1, 2, 0).cpu().numpy()[..., ::-1])
            ax.axis('off')
            ax.set_title(f"Input {j + 1}")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the wspace and hspace to fine-tune the gaps
    wandb.log({"viz_{}".format(step_name): wandb.Image(fig)}, step=step)
    plt.close(fig)

def viz_tactile_figure(ground_truth_tactile, predicted_frames_tactile, config, step, step_name):        # we want to plot each of the 16 features in a 4x4 grid over the ts frames:
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    if len(predicted_frames_tactile.shape) > 2: predicted_frames_tactile = predicted_frames_tactile.squeeze()

    x_pred = predicted_frames_tactile[:, :16].cpu().numpy()
    x_gt = ground_truth_tactile[0][:, :16].cpu().numpy()
    y_pred = predicted_frames_tactile[:, 16:32].cpu().numpy()
    y_gt = ground_truth_tactile[0][:, 16:32].cpu().numpy()
    z_pred = predicted_frames_tactile[:, 32:].cpu().numpy()
    z_gt = ground_truth_tactile[0][:, 32:].cpu().numpy()

    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.plot(x_gt[:, i], label="x gt", linestyle="--", color="red")
        ax.plot(x_pred[:, i], label="x pred", color="red")
        ax.plot(y_gt[:, i], label="y gt", linestyle="--", color="green")
        ax.plot(y_pred[:, i], label="y pred", color="green")
        ax.plot(z_gt[:, i], label="z gt", linestyle="--", color="blue")
        ax.plot(z_pred[:, i], label="z pred", color="blue")
        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, config.prediction_horizon, 2))
        ax.set_title(f"Feature {i}")
        # set x ticks every 10 points
        ax.set_xticks(range(0, config.prediction_horizon, 5))

    fig.legend(["x gt", "x pred", "y gt", "y pred", "z gt", "z pred"], loc='lower center', ncol=3)
    plt.tight_layout()
    wandb.log({"viz_tactile_{}".format(step_name): wandb.Image(fig)}, step=step)
    plt.close(fig)

def viz_rollout_losses(loss_sequences_combined, loss_sequences_image, loss_sequences_tactile, config, step):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if config.model_type == "transformer":
        if config.image and config.tactile:
            loss_sequences_combined = np.array(loss_sequences_combined)
            mean_loss = np.mean(loss_sequences_combined, axis=0)
            std_loss = np.std(loss_sequences_combined, axis=0)
            ax.plot(mean_loss, label="combined loss", color="red")
        if config.image:
            loss_sequences_image = np.array(loss_sequences_image)
            mean_loss = np.mean(loss_sequences_image, axis=0)
            std_loss = np.std(loss_sequences_image, axis=0)
            ax.plot(mean_loss, label="image loss", color="green")
        if config.tactile:
            loss_sequences_tactile = np.array(loss_sequences_tactile)
            mean_loss = np.mean(loss_sequences_tactile, axis=0)
            std_loss = np.std(loss_sequences_tactile, axis=0)
            ax.plot(mean_loss, label="tactile loss", color="blue")        
    elif config.model_type == "SVG":
        if config.image and config.tactile:
            # the actual combined loss in the SVG model is the sum of the combined and tactile losses
            loss_sequences_combined = np.array(loss_sequences_combined) + torch.tensor(loss_sequences_tactile).cpu().numpy()
            mean_loss = np.mean(loss_sequences_combined, axis=0)
            std_loss = np.std(loss_sequences_combined, axis=0)
            ax.plot(mean_loss, label="combined loss", color="red")
        if config.image:
            loss_sequences_combined = np.array(loss_sequences_combined)
            mean_loss = np.mean(loss_sequences_combined, axis=0)
            std_loss = np.std(loss_sequences_combined, axis=0)
            ax.plot(mean_loss, label="image loss", color="green")
            loss_sequences_image = np.array(loss_sequences_image)
            mean_loss = np.mean(loss_sequences_image, axis=0)
            std_loss = np.std(loss_sequences_image, axis=0)
            ax.plot(mean_loss, label="image prior loss", color="purple")
        if config.tactile:
            loss_sequences_tactile = torch.tensor(loss_sequences_tactile).cpu().numpy()
            mean_loss = np.mean(loss_sequences_tactile, axis=0)
            std_loss = np.std(loss_sequences_tactile, axis=0)
            ax.plot(mean_loss, label="tactile loss", color="blue")        

    ax.fill_between(range(config.prediction_horizon), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    ax.set_title("Rollout Loss")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Loss MAE")
    ax.legend()
    wandb.log({"combined_loss_over_time": wandb.Image(fig)}, step=step)
    plt.close(fig)

def visualise_step(images, name):
    import imageio
    imageio.mimsave(name + ".gif", images, duration=0.1)

def viz_tactile_histogram(tactile_data):
    # plot histogram of float point values with matplotlib
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    name = ["Shear x", "Shear y", "Normal z"]
    for i in range(3):
        ax[i].hist(tactile_data[:, i, :].flatten(), bins=200)
        ax[i].set_title(f"{name[i]}")
        ax[i].set_xlabel("Force value")
        ax[i].set_ylabel("Frequency")
    plt.tight_layout()
    wandb.log({"tactile_histogram": wandb.Image(plt)}, step=0)
    plt.close()

###########################
#
# Training and Validation functions
#
###########################
def format_and_run_batch(batch, config, model, criterion, timer, horizon_rollout, repeatable_infill=False, eval=False):
    image_context, image_predict, tactile_context, tactile_predict, robot_data = None, None, None, None, None
    if horizon_rollout:
        if config.image:
            image_context = batch[1][:, :config.context_length, ...].to(config.device)    # take all but the last image          shape = [bs, c, 64, 64, 3])
            image_predict = batch[1][:,  config.context_length:, ...].to(config.device)   # take just the last image             shape = [bs, p,     64, 64, 3])
            if config.infill_patches:
                if repeatable_infill:
                    x = config.repeatable_infil_x_pos
                    y = config.repeatable_infil_y_pos
                    infill_patch_size = config.repeatable_infil_patch_size
                else:
                    infill_patch_size = np.random.randint(config.min_infill_patch_size, config.max_infill_patch_size)
                    x = np.random.randint(0, config.image_height - infill_patch_size)
                    y = np.random.randint(0, config.image_width  - infill_patch_size)
                image_context[:, :, :, x:x+infill_patch_size, y:y+infill_patch_size] = 0.0
        if config.action:
            robot_data    = batch[0].to(config.device)                                          # take the full sequence of robot data shape = [bs, c+p,   6])       
        if config.tactile:
            tactile_context = batch[2][:, :config.context_length, ...].to(config.device)
            tactile_predict = batch[2][:,  config.context_length:, ...].to(config.device)
    else:
        if config.image:
            image_context = batch[1][:, :-1, ...].to(config.device)    # take all but the last image          shape = [bs, c+p-1, 64, 64, 3])
            image_predict = batch[1][:,  1:, ...].to(config.device)    # take just the last image             shape = [bs, 1,     64, 64, 3])
            if config.infill_patches:
                if repeatable_infill:
                    x = config.repeatable_infil_x_pos
                    y = config.repeatable_infil_y_pos
                    infill_patch_size = config.repeatable_infil_patch_size
                else:
                    infill_patch_size = np.random.randint(config.min_infill_patch_size, config.max_infill_patch_size)
                    x = np.random.randint(0, config.image_height - infill_patch_size)
                    y = np.random.randint(0, config.image_width  - infill_patch_size)
                image_context[:, :, :, x:x+infill_patch_size, y:y+infill_patch_size] = 0.0
        if config.action:
            robot_data    = batch[0].to(config.device)                   # take the full sequence of robot data shape = [bs, c+p,   6])
        if config.tactile:
            tactile_context = batch[2][:, :-1, ...].to(config.device)    # take all but the last image          shape = [bs, c+p-1, 48])
            tactile_predict = batch[2][:,  1:, ...].to(config.device)    # take just the last image             shape = [bs, 1,     48])

    # run the model
    if horizon_rollout:
        (rollout_image_prediction, image_groundtruth, rollout_tactile_prediction, tactile_groundtruth, image_losses, tactile_losses, combined_total_loss, 
        loss_sequence_image, loss_sequence_tactile, loss_sequence_combined) = rollout_sequence(image_context, image_predict, robot_data, tactile_context, tactile_predict, config, model, criterion) # TODO: add tactile data
        return rollout_image_prediction, image_groundtruth, rollout_tactile_prediction, tactile_groundtruth, image_losses, tactile_losses, combined_total_loss, loss_sequence_image, loss_sequence_tactile, loss_sequence_combined, image_context
    else:
        with timer("train"): pred_image, pred_tactile, total_loss, loss, tactile_loss = model(image_context, targets=image_predict, actions=robot_data, tactiles=tactile_context, tactile_targets=tactile_predict, test=eval)        # forward pass
        return pred_image, image_predict, pred_tactile, tactile_predict, total_loss, loss, tactile_loss, image_context


def rollout_sequence(image_context, image_groundtruth, robot_data, tactile_context, tactile_groundtruth, config, model, criterion):
    predicted_image_sequence = []
    predicted_tactile_sequence = []
    image_predict, robot_data_sequence, tactile_predict = None, None, None
    rollout_image_prediction, rollout_tactile_prediction = None, None
    image_losses, tactile_losses, combined_total_loss = None, None, None
    robot_data_sequence = None
    full_tactile_sequence = None

    loss_sequence_combined = []
    loss_sequence_image = []
    loss_sequence_tactile = []

    if config.image:   full_image_sequence = torch.cat([image_context, image_groundtruth], dim=1)
    if config.tactile: full_tactile_sequence = torch.cat([tactile_context, tactile_groundtruth], dim=1)

    if config.model_type == "transformer":
        for j in range(config.prediction_horizon):
            if config.image:    image_predict = full_image_sequence[:, j+1:j + config.context_length + 1, ...]
            if config.action:   robot_data_sequence = robot_data[:, j:j + config.context_length + 1, ...]
            if config.tactile:  tactile_predict = full_tactile_sequence[:, j+1:j + config.context_length + 1, ...]

            pred_image, pred_tactile, total_loss, loss, tactile_loss = model(image_context, targets=image_predict, actions=robot_data_sequence, tactiles=tactile_context, tactile_targets=tactile_predict)

            if config.image:
                image_context = torch.cat([image_context[:, 1:], pred_image[:, -1:, ]], dim=1)
                predicted_image_sequence.append(pred_image[0, -1, ...])
                image_loss_timestep_x   = criterion(image_predict[0][-1], pred_image[0][-1])
                loss_sequence_image.append(image_loss_timestep_x.item())
            if config.tactile:
                tactile_context = torch.cat([tactile_context[:, 1:], pred_tactile[:, -1:, ]], dim=1)
                predicted_tactile_sequence.append(pred_tactile[0, -1, ...])
                tactile_loss_timestep_x = criterion(tactile_predict[0][-1], pred_tactile[0][-1])
                loss_sequence_tactile.append(tactile_loss_timestep_x.item())

            if config.image and config.tactile:
                loss_sequence_combined.append(image_loss_timestep_x.item() + tactile_loss_timestep_x.item())

        if config.image:
            rollout_image_prediction = torch.stack(predicted_image_sequence, dim=0)
            image_losses = criterion(image_groundtruth[0], rollout_image_prediction)
        if config.tactile:
            rollout_tactile_prediction = torch.stack(predicted_tactile_sequence, dim=0)
            tactile_losses = criterion(tactile_groundtruth[0], rollout_tactile_prediction)
        if config.image and config.tactile:
            combined_total_loss = image_losses + tactile_losses
        if combined_total_loss == None:
            combined_total_loss = image_losses if config.image else tactile_losses

    elif config.model_type == "SVG":
        # outputs_scene, outputs_tactile, mae_scene_loss, mae_scene_prior_loss, mae_tactile_loss, mae_scene_list, kld_scene_list, mae_tactile_list  # this is the actual output of the model - we keep the names the same for consistency with the other models
        rollout_image_prediction, rollout_tactile_prediction, combined_total_loss, image_losses, tactile_losses, loss_sequence_combined, loss_sequence_image, loss_sequence_tactile  = model(image_context, targets=image_groundtruth, actions=robot_data, tactiles=full_tactile_sequence, tactile_targets=tactile_groundtruth, rollout=True)

    return (rollout_image_prediction, image_groundtruth, rollout_tactile_prediction, tactile_groundtruth, image_losses, tactile_losses, combined_total_loss, 
            loss_sequence_image, loss_sequence_tactile, loss_sequence_combined)

def validate_action_space(robot_state, config):
    pass

###########################
#
# Very Util util funcitons haha
#
###########################

def save_model(model, name, config, wandb_id):
    os.makedirs(config.save_dir + "/" + config.model_name + "/" + wandb_id + "/", exist_ok=True)
    save_path = config.save_dir + "/" + config.model_name + "/" + wandb_id + "/" + name + ".pth"
    if config.model_type == "transformer":    torch.save(model.state_dict(), save_path)
    elif config.model_type == "lstm":         model.save_model(save_path)

def wandb_log(info, step):
    wandb.log(flatten_dict(info, sep="/"), step=step)

def format_name_with_config(name, config):
    """Formats a name string with a config dict.

    Formatting keys may be specified as {key} or {full_path_to_key_with_underscores}.

    Example:
        name = "model_{model_type}_{model_size}"
        config = {"model_type": "transformer", "model_size": "small"}
        format_name_with_config(name, config) -> "model_transformer_small"
    """
    config_flat = flax.traverse_util.flatten_dict(config, sep="_")
    config_final = {k.split("_")[-1]: v for k, v in config_flat.items()}
    format_dict = {**config_final, **config_flat}
    return name.format(**format_dict)

class Timer:
    """
    Timer utility. Usage:

        timer = Timer()
        with timer("foo"):
            do_something()

        timer.tick("bar")
        do_something_else()
        timer.tock("bar")

        timer.get_average_times() -> {"foo": 0.1, "bar": 0.2}
    """

    def __init__(self):
        self.reset()

    @contextmanager
    def __call__(self, key):
        self.tick(key)
        try:
            yield None
        finally:
            self.tock(key)

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def get_average_times(self, reset=True):
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret

def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
