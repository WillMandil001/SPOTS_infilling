import os
import wandb
import torch
import joblib
import datetime
import itertools
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm
from train_utils import format_name_with_config, Timer, tree_map, VisionTactileDataset, save_model, wandb_log, val_step, viz_step, train_step

# models 
from model import VPGPT

# data loading and processing
from ml_collections import config_flags
from torch.utils.data import Dataset, DataLoader

# data logging and visualization
from absl import app, flags, logging
from flax.traverse_util import flatten_dict

# TODO:
# [ ] Add the dataset to the GPU straight away instead of moving it in the training loop

###########################
# Set up the Flags for the training
###########################
FLAGS = flags.FLAGS
default_config_file = os.path.dirname(__file__) + "/config/config_base_model.py"
config_flags.DEFINE_config_file("config", default_config_file, "File path to the training hyperparameter configuration.", lock_config=False)
flags.DEFINE_bool   ("debug",        False,                                                                            "Debug config (no wandb logging)")


def main(_):
    ###########################
    # Setup WandB
    ###########################
    wandb.login()

    name = format_name_with_config(FLAGS.config.experiment_name, FLAGS.config.to_dict() )
    wandb_id = "{name}_{time}".format( name=FLAGS.config.experiment_name, time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    wandb.init( config=FLAGS.config.to_dict(), id=wandb_id, name=FLAGS.config.experiment_name, mode="disabled" if FLAGS.debug else None, **FLAGS.config.wandb)

    # set run directory to save the logs    
    logging.info("Wandb logs saved to %s", wandb.run.dir)
    logging.info("Wandb url: %s", wandb.run.get_url())

    if FLAGS.debug:
        FLAGS.config.eval_interval = 10

    ###########################
    # Load the dataset  | load the tfrecords RLDS dataset saved locally at: /home/wmandil/tensorflow_datasets/robot_pushing_dataset/1.0.0
    ###########################
    train_dataset = VisionTactileDataset(config=FLAGS, map_file=FLAGS.config.dataset_train_dir + "map.npy", context_len=FLAGS.config.context_length,  prediction_horizon=1, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.config.batch_size, shuffle=False, num_workers=FLAGS.config.num_workers)

    val_dataset = VisionTactileDataset(config=FLAGS, map_file=FLAGS.config.dataset_val_dir + "map.npy", context_len=FLAGS.config.context_length, prediction_horizon=1, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=FLAGS.config.batch_size, shuffle=False, num_workers=FLAGS.config.num_workers)

    viz_dataset = VisionTactileDataset(config=FLAGS, map_file=FLAGS.config.dataset_val_dir + "map.npy", context_len=FLAGS.config.context_length, prediction_horizon=FLAGS.config.prediction_horizon, train=False)
    viz_dataloader = DataLoader(viz_dataset, batch_size=1, shuffle=False, num_workers=FLAGS.config.num_workers)

    ###########################
    # Load the model and optimizer
    ###########################
    model = VPGPT(FLAGS.config.model_config).to(FLAGS.config.device)
    scaler = torch.cuda.amp.GradScaler(enabled=(FLAGS.config.dtype == 'float16'))
    optimizer = model.configure_optimizers(FLAGS.config.weight_decay, FLAGS.config.learning_rate, (FLAGS.config.beta1, FLAGS.config.beta2), FLAGS.config.device)

    if FLAGS.config.criterion == "MAE":    criterion = nn.L1Loss()
    elif FLAGS.config.criterion == "MSE":  criterion = nn.MSELoss()

    ###########################
    # Save all metadata to wandb
    ###########################
    save_dir = os.path.join( FLAGS.config.save_dir, FLAGS.config.wandb.project, FLAGS.config.wandb.group or "", wandb_id)
    wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
    logging.info("Saving to %s", save_dir)

    example_batch = next(iter(train_dataloader))
    example_batch_spec = {k: v.shape for k, v in zip(["robot", "image", "tactile"], example_batch)}
    wandb.config.update(dict(example_batch_spec=example_batch_spec), allow_val_change=True)

    # save the dataset sizes
    wandb.config.update(dict(train_dataset_size=len(train_dataset), val_dataset_size=len(val_dataset), viz_dataset_size=len(viz_dataset)), allow_val_change=True)

    # save the model config
    model_config = {k: v for k, v in FLAGS.config.model_config.items() if k != "features"}
    wandb.config.update(dict(model_config=model_config), allow_val_change=True)

    # save the model attention mask
    plot = model.get_attention_mask()  # returns a matplotlib plot of the attention mask
    wandb.log({"attention_mask": wandb.Image(plot)}, step=0)
    
    ###########################
    # training loop
    ###########################
    logging.info(f" --- starting training loop")

    model.train()
    timer = Timer()
    step = 0
    with tqdm(total=FLAGS.config.num_steps, dynamic_ncols=True) as pbar:
        timer.tick("batch_gen")
        timer.tick("total")
        while step < FLAGS.config.num_steps:
            for batch in train_dataloader:
                timer.tock("batch_gen")
                with timer("format and train"):   update_info = train_step(batch, FLAGS, scaler, model, optimizer, criterion, timer)
                timer.tock("total")

                if (step + 1) % FLAGS.config.log_interval == 0:
                    wandb_log({"training": update_info, "timer": timer.get_average_times()}, step=step)

                if (step + 1) % FLAGS.config.eval_interval == 0:
                    with timer("val"):        
                        val_update_info = val_step(step, FLAGS, model, criterion, val_dataloader, timer)
                        wandb_log(val_update_info, step=step)
                    with timer("visualize"):  
                        viz_update_info = viz_step(step, FLAGS, model, criterion, viz_dataloader, timer)
                        wandb_log(viz_update_info, step=step)
                    model.train()

                if (step + 1) % FLAGS.config.save_interval == 0:
                    save_model(model, "model_step_{i}".format(i=step), FLAGS, wandb_id)

                step += 1
                pbar.update(1)
                timer.tick("batch_gen")
                timer.tick("total")

    save_model(model, "model_final")

if __name__ == "__main__":
    app.run(main)
