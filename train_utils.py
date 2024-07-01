

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
import logging
import hashlib
import torch
import json
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import dlimp as dl
# from absl import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Mapping, Union

import joblib
import itertools
import torch.nn as nn
import matplotlib.pyplot as plt
from flax.traverse_util import flatten_dict

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class VisionTactileDataset(Dataset):
    def __init__(self, config, map_file, context_len=10, prediction_horizon=10, train=True, wandb_id= ""):
        self.flags = config
        self.train = train
        self.map_file = map_file
        self.context_len = context_len
        self.wandb_id = wandb_id

        if train == True:  self.prediction_horizon = 1
        else:              self.prediction_horizon = prediction_horizon

        self.map_data = np.load(self.map_file, allow_pickle=True)
        if config.debug:
            self.map_data = self.map_data[0:10]
        self.build_dataset()

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        if self.flags.config.pre_load_data:
            start_index = self.sample_index_list[idx]
            robot_state, image_data, tactile_data = [], [], []
            for i in range(0, (self.context_len + self.prediction_horizon)*self.flags.config.sample_rate, self.flags.config.sample_rate):
                step_data = self.data[start_index + i]
                if self.flags.config.action:      robot_state.append(step_data[0])
                if self.flags.config.image:       image_data.append(step_data[1].astype(np.float32) / 255)
                if self.flags.config.tactile:     tactile_data.append(step_data[2].flatten())
        else:
            steps = self.sequences[idx:idx + self.context_len + self.prediction_horizon]  # TODO wont work with sample_rate!
            robot_state, image_data, tactile_data  = [], [], []
            for save_name in steps:
                step_data = np.load(save_name, allow_pickle=True)
                if self.flags.config.action:      robot_state.append(step_data[()]["state"].astype(np.float32) / 255)
                if self.flags.config.image:       image_data.append(step_data[()]['image'].astype(np.float32) / 255)
                if self.flags.config.tactile:     tactile_data.append(step_data[()]['tactile'].astype(np.float32))

        if self.flags.config.action:   robot_state = np.stack(robot_state, axis=0)         # shape is robot=[c+p, bs, 6]
        if self.flags.config.image:    image_data  = np.stack(image_data, axis=0)     # shape is images=[c+p, bs, 64,64,3] we need to flip the channels so that its [bs, c+p, 3, 64, 64] (done in the return)
        if self.flags.config.tactile:  tactile_data = np.stack(tactile_data, axis=0)  # shape is tactile=[c+p, bs, 48]

        return torch.tensor(robot_state), torch.tensor(image_data).permute(0, 3, 1, 2) ,torch.tensor(tactile_data)

    def build_dataset(self):
        self.total_sequences = 0
        self.sequences = []
        for episode in self.map_data:
            episode_length = episode['episode_length']
            valid_sequences = episode_length - ((self.context_len + self.prediction_horizon - 1)*self.flags.config.sample_rate)
            if valid_sequences > 0:
                self.total_sequences += valid_sequences
                self.sequences += episode['step_save_name_list'][self.context_len + self.prediction_horizon - 1:]  #  TODO not needed for pre-loaded datasets and needs a fix for sample_rate stuff 

        if self.flags.config.pre_load_data:
            self.data = []
            self.sample_index_list = []
            current_index = 0
            for episode in tqdm(self.map_data, desc="Loading data", dynamic_ncols=True):
                episode_length = episode['episode_length']
                for step_num, save_name in enumerate(episode['step_save_name_list']):
                    save_name = save_name.replace(self.flags.config.to_replace, self.flags.config.replace_with)            # overwrite location if it has changed:
                    step_data = np.load(save_name, allow_pickle=True)
                    robot_state  = step_data[()]["state"]
                    image_data   = step_data[()]['image']
                    tactile_data = step_data[()]['tactile']
                    if episode_length - step_num >= (self.context_len + self.prediction_horizon - 1)*self.flags.config.sample_rate:
                        self.sample_index_list += [current_index]
                    current_index += 1
                    self.data.append([robot_state, image_data, tactile_data])

        if self.flags.config.scale_tactile_tactile:
            tactile_data = np.array([i[2] for i in self.data])
            robot_state_data  = np.array([i[0] for i in self.data])

            # Create MinMaxScaler instances for each axis
            if self.train == True:
                self.tactile_scaler_x   = MinMaxScaler(feature_range=(0, 1))
                self.tactile_scaler_y   = MinMaxScaler(feature_range=(0, 1))
                self.tactile_scaler_z   = MinMaxScaler(feature_range=(0, 1))
                self.robot_state_norm   = StandardScaler()
                self.robot_state_scaler = MinMaxScaler(feature_range=(0, 1))
            else: # load the scalars from the save_dir:
                self.tactile_scaler_x   = joblib.load(os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "tactile_scaler_x.pkl"))
                self.tactile_scaler_y   = joblib.load(os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "tactile_scaler_y.pkl"))
                self.tactile_scaler_z   = joblib.load(os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "tactile_scaler_z.pkl"))
                self.robot_state_norm   = joblib.load(os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "robot_state_norm.pkl"))
                self.robot_state_scaler = joblib.load(os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "robot_state_scaler.pkl"))

            # Fit the scalers on the corresponding slices of the tactile data
            self.tactile_scaler_x.fit(tactile_data[:, 0, :])
            self.tactile_scaler_y.fit(tactile_data[:, 1, :])
            self.tactile_scaler_z.fit(tactile_data[:, 2, :])

            # Transform the data (tactile)
            tactile_data[:, 0, :] = self.tactile_scaler_x.transform(tactile_data[:, 0, :])
            tactile_data[:, 1, :] = self.tactile_scaler_y.transform(tactile_data[:, 1, :])
            tactile_data[:, 2, :] = self.tactile_scaler_z.transform(tactile_data[:, 2, :])

            # normalise then scale the data (action) - we have to do this in two steps
            self.robot_state_norm.fit(robot_state_data)
            robot_state_data      = self.robot_state_norm.transform(robot_state_data)
            self.robot_state_scaler.fit(robot_state_data)
            robot_state_data      = self.robot_state_scaler.transform(robot_state_data)

            viz_tactile_histogram(tactile_data)

            for i in range(len(self.data)):
                self.data[i][2] = tactile_data[i]
                self.data[i][0] = robot_state_data[i]

            if self.train:
                os.makedirs(os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id), exist_ok=True)
                joblib.dump(self.tactile_scaler_x,   os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "tactile_scaler_x.pkl"))
                joblib.dump(self.tactile_scaler_y,   os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "tactile_scaler_y.pkl"))
                joblib.dump(self.tactile_scaler_z,   os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "tactile_scaler_z.pkl"))
                joblib.dump(self.robot_state_norm,   os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "robot_state_norm.pkl"))
                joblib.dump(self.robot_state_scaler, os.path.join(self.flags.config.save_dir, self.flags.config.model_name, self.wandb_id, "robot_state_scaler.pkl"))

###########################
#
# Vizualisation functions
#
###########################

def viz_image_figure(ground_truth, predicted_frames, config, step):
    fig, axes = plt.subplots(2, config.prediction_horizon, figsize=(config.prediction_horizon * 3, 6))
    for j in range(config.prediction_horizon):
        ax = axes[0, j]
        ax.imshow(ground_truth[0][j].permute(1, 2, 0).cpu().numpy()[..., ::-1])
        ax.axis('off')
        ax.set_title(f"GT {j+1}")
        ax = axes[1, j]
        ax.imshow(predicted_frames[j].permute(1, 2, 0).cpu().numpy()[..., ::-1])
        ax.axis('off')
        ax.set_title(f"Pred {j+1}")
    plt.tight_layout()
    wandb.log({"viz_{}".format(i): wandb.Image(fig)}, step=step)
    plt.close(fig)

def viz_tactile_figure(ground_truth_tactile, predicted_frames_tactile, config, step):        # we want to plot each of the 16 features in a 4x4 grid over the ts frames:
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
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
    fig.legend(["x gt", "x pred", "y gt", "y pred", "z gt", "z pred"], loc='lower center', ncol=3)
    plt.tight_layout()
    wandb.log({"viz_tactile_{}".format(i): wandb.Image(fig)}, step=step)
    plt.close(fig)

def viz_rollout_losses(loss_sequences_combined, loss_sequences_image, loss_sequences_tactile, config, step):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
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

    ax.fill_between(range(config.prediction_horizon), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    ax.set_title("Combined loss over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Loss")
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
def train_step(batch, flags, scaler, model, optimizer, criterion, timer):
    pred_image, image_predict, pred_tactile, tactile_predict, total_loss, loss, tactile_loss = format_and_run_batch(batch, flags.config, model, criterion, timer, horizon_rollout=False)
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    update_info = {"grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": total_loss.item()}
    if flags.config.image:    update_info["image_loss"] = loss.item()
    if flags.config.tactile:  update_info["tactile_loss"] = tactile_loss.item()    

    return update_info

def val_step(step, flags, model, criterion, val_dataloader, timer):
    val_metrics = {}
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_dataloader):
            pred_image, image_predict, pred_tactile, tactile_predict, total_loss, loss, tactile_loss = format_and_run_batch(batch, flags.config, model, criterion, timer, horizon_rollout=False)
            val_metrics["validation_loss"] = val_metrics.get("loss", 0) + total_loss.item()
            if flags.config.tactile:  val_metrics["tactile_loss"] = val_metrics.get("tactile_loss", 0) + tactile_loss.item()
            if flags.config.image:    val_metrics["image_loss"] = val_metrics.get("image_loss", 0) + loss.item()
    val_metrics = tree_map(lambda x: x / (step + 1), val_metrics)
    return val_metrics

def viz_step(step, flags, model, criterion, viz_dataloader, timer):
    with torch.no_grad():
        model.eval()
        combined_losses, image_loss_list, tactile_loss_list = [], [], []
        loss_sequences_image, loss_sequences_tactile, loss_sequences_combined = [], [], []
        for i, batch in enumerate(viz_dataloader):
            if i not in flags.config.viz_steps:  continue
            (rollout_image_prediction, image_groundtruth, rollout_tactile_prediction, tactile_groundtruth, image_losses, tactile_losses, combined_total_loss, 
            loss_sequence_image, loss_sequence_tactile, loss_sequence_combined) = format_and_run_batch(batch, flags.config, model, criterion, timer, horizon_rollout=True)
            if flags.config.image:
                viz_image_figure(image_groundtruth, rollout_image_prediction, flags.config, step)
                image_loss_list.append(image_losses.item())
                loss_sequences_image.append(loss_sequence_image)
            if flags.config.tactile:
                viz_tactile_figure(tactile_groundtruth, rollout_tactile_prediction, flags.config, step)
                tactile_loss_list.append(tactile_losses.item())
                loss_sequences_tactile.append(loss_sequence_tactile)
            if flags.config.image and flags.config.tactile:
                combined_losses.append(combined_total_loss.item())
                loss_sequences_combined.append(loss_sequence_combined)

    viz_metrics = {}
    if flags.config.image and flags.config.tactile:  viz_metrics["rollout combined loss {}".format(flags.config.prediction_horizon)] = np.mean(combined_losses)
    if flags.config.image:                           viz_metrics["rollout image loss {}".format(flags.config.prediction_horizon)]    = np.mean(image_loss_list)
    if flags.config.tactile:                         viz_metrics["rollout tactile loss {}".format(flags.config.prediction_horizon)]  = np.mean(tactile_loss_list)

    viz_rollout_losses(loss_sequences_combined, loss_sequences_image, loss_sequences_tactile, flags.config, step)
    return viz_metrics

def format_and_run_batch(batch, config, model, criterion, timer, horizon_rollout):
    image_context, image_predict, tactile_context, tactile_predict, robot_data = None, None, None, None, None
    if horizon_rollout:
        if config.image:
            image_context = batch[1][:, :config.context_length, ...].to(config.device)    # take all but the last image          shape = [bs, c, 64, 64, 3])
            image_predict = batch[1][:,  config.context_length:, ...].to(config.device)   # take just the last image             shape = [bs, p,     64, 64, 3])
            if config.infill_patches:
                x = np.random.randint(0, config.image_height - config.patch_size)
                y = np.random.randint(0, config.image_width  - config.patch_size)
                image_context[:, :, :, x:x+config.patch_size, y:y+config.patch_size] = 0.0
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
                x = np.random.randint(0, config.image_height - config.patch_size)
                y = np.random.randint(0, config.image_width  - config.patch_size)
                image_context[:, :, :, x:x+config.patch_size, y:y+config.patch_size] = 0.0
        if config.action:
            robot_data    = batch[0].to(config.device)                   # take the full sequence of robot data shape = [bs, c+p,   6])
        if config.tactile:
            tactile_context = batch[2][:, :-1, ...].to(config.device)    # take all but the last image          shape = [bs, c+p-1, 48])
            tactile_predict = batch[2][:,  1:, ...].to(config.device)    # take just the last image             shape = [bs, 1,     48])

    # run the model
    if horizon_rollout:
        (rollout_image_prediction, image_groundtruth, rollout_tactile_prediction, tactile_groundtruth, image_losses, tactile_losses, combined_total_loss, 
        loss_sequence_image, loss_sequence_tactile, loss_sequence_combined) = rollout_sequence(image_context, image_predict, robot_data, tactile_context, tactile_predict, config, model, criterion) # TODO: add tactile data
        return rollout_image_prediction, image_groundtruth, rollout_tactile_prediction, tactile_groundtruth, image_losses, tactile_losses, combined_total_loss, loss_sequence_image, loss_sequence_tactile, loss_sequence_combined
    else:
        with timer("train"): pred_image, pred_tactile, total_loss, loss, tactile_loss = model(image_context, targets=image_predict, actions=robot_data, tactiles=tactile_context, tactile_targets=tactile_predict)        # forward pass
        return pred_image, image_predict, pred_tactile, tactile_predict, total_loss, loss, tactile_loss


def rollout_sequence(image_context, image_groundtruth, robot_data, tactile_context, tactile_groundtruth, config, model, criterion):
    predicted_image_sequence = []
    predicted_tactile_sequence = []
    rollout_image_prediction, rollout_tactile_prediction = None, None
    image_losses, tactile_losses, combined_total_loss = None, None, None
    robot_data_sequence = None

    loss_sequence_combined = []
    loss_sequence_image = []
    loss_sequence_tactile = []

    if config.image:   full_image_sequence = torch.cat([image_context, image_groundtruth], dim=1)
    if config.tactile: full_tactile_sequence = torch.cat([tactile_context, tactile_groundtruth], dim=1)

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

    return (rollout_image_prediction, image_groundtruth, rollout_tactile_prediction, tactile_groundtruth, image_losses, tactile_losses, combined_total_loss, 
            loss_sequence_image, loss_sequence_tactile, loss_sequence_combined)


###########################
#
# Very Util util funcitons haha
#
###########################


def save_model(model, name, config, wandb_id):
    os.makedirs(config.save_dir + "/" + config.model_name + "/" + wandb_id + "/", exist_ok=True)
    torch.save(model.state_dict(), config.save_dir + "/" + config.model_name + "/" + wandb_id + "/" + name + ".pth")  # save the model

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
