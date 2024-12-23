import os
import wandb
import torch
import joblib
import datetime
import numpy as np
import torch.nn as nn

# training utilities
import train_utils_gelsight as train_utils

from config.config_base_model_gelsight import Config, model_config_builder_transformer, model_config_builder_actp, model_config_builder_svg

# models
from model_set_gel_sight.transformer import VPGPT

# data loading and processing
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# data logging and visualization
from absl import app, logging, flags

###########################
# Set up the Flags for the training
###########################
FLAGS = flags.FLAGS

# experiment / run flags
# flags.DEFINE_string ('model_name',               "SVG-ACTP",     'write the model name here (VGPT, AC-VGPT, AC-VTGPT, SVG, SVG-ACTP, SVG-ACTP-SOP)')
# flags.DEFINE_string ('model_type',               "SVG",  'Set the type of model you are going to use (transformer, SVG, ACTP)')
flags.DEFINE_string ('model_name',               "VTGPT",     'write the model name here (VGPT, AC-VGPT, AC-VTGPT, SVG, SVG-ACTP, SVG-ACTP-SOP)')
flags.DEFINE_string ('model_type',               "transformer",  'Set the type of model you are going to use (transformer, SVG, ACTP)')
flags.DEFINE_string ('test_version',             "testing...",   'just a filler name for logging - set to vXX or testXXX')
flags.DEFINE_boolean('train_infill',             True,          'Whether to infill or not')
flags.DEFINE_boolean('test_infill',              True,          'Whether to infill or not')
flags.DEFINE_boolean('train_tactile_infill',     True,          'Whether to infill or not')
flags.DEFINE_boolean('test_tactile_infill',      False,          'Whether to infill or not')
flags.DEFINE_boolean('cluster',                  False,          'Whether or not to run on the cluster')

# training flags
flags.DEFINE_integer('num_steps',                0,        'set to 0 to use the configs num_steps') 

# model specific flags
flags.DEFINE_boolean('use_all_tactile_samples',  False,       'whether to use all the tactile frames between the under sampled sequences')

# Pre-training flags
flags.DEFINE_boolean('pretrained',               False,       '')
flags.DEFINE_boolean('pretrained_enc',           False,       '')
flags.DEFINE_boolean('pretrained_enc_frozen',    False,       '')
flags.DEFINE_boolean('pretrained_ac_enc',        False,       '')
flags.DEFINE_boolean('pretrained_ac_enc_frozen', False,       '')
flags.DEFINE_boolean('pretrained_tok',           False,       '')
flags.DEFINE_boolean('pretrained_tok_frozen',    False,       '')
flags.DEFINE_boolean('pretrained_dec',           False,       '')
flags.DEFINE_boolean('pretrained_dec_frozen',    False,       '')
flags.DEFINE_string ('pretrained_model_path',   "/home/wmandil/robotics/saved_models/VGPT-infill/v4 - VGPT-infill_20240717_082440/model_final.pth",  '')
flags.DEFINE_string ('pretrained_config_path',  "/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_082440-v4 - VGPT-infill_20240717_082440/files/config.yaml",  '')

###########################
# Build the dataset and dataloader
###########################
class VisionTactileDataset(Dataset):
    def __init__(self, config, map_file, context_len, prediction_horizon, train=True, validate=False, wandb_id= "", test=False, vis=False):
        self.config = config
        self.train = train
        self.map_file = map_file
        self.context_len = context_len
        self.wandb_id = wandb_id

        if (train == True or validate==True) and self.config.model_type=="transformer": self.prediction_horizon = 1  # we want it always to be 1! as we dont rollout the transformer in training.
        else:                                                                           self.prediction_horizon = prediction_horizon

        print("self.prediction_horizon: ", self.prediction_horizon)

        self.map_data = np.load(self.map_file, allow_pickle=True)

        if config.debug:
            self.map_data = self.map_data[:10]
        elif train == True:
            self.map_data = self.map_data[:-300]
        elif test == True:
            self.map_data = self.map_data[-300:][::10]
        elif train == False and test == False:
            self.map_data = self.map_data[-300:][::10]

        self.build_dataset()

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_index = self.sample_index_list[idx]
        image_data, tactile_data = [], []
        for i in range(0, (self.context_len + self.prediction_horizon)*self.config.sample_rate, self.config.sample_rate):
            if self.config.pre_load_data:
                step_data = self.data[start_index + i]
                if self.config.image:       image_data.append(step_data[0].astype(np.float32) / 255)
                if self.config.tactile:     tactile_data.append(step_data[1].astype(np.float32) / 255)
            else:
                step_data = np.load(self.data[start_index + i], allow_pickle=True)
                if self.config.image:       image_data.append(step_data[()]['image'].transpose(2, 0, 1).astype(np.float32) / 255)
                if self.config.tactile:     tactile_data.append(step_data[()]['tactile'].transpose(2, 0, 1).astype(np.float32) / 255)

        if self.config.image:    image_data   = np.stack(image_data, axis=0)     # shape is images=[c+p, bs, 64,64,3] we need to flip the channels so that its [bs, c+p, 3, 64, 64] (done in the return)
        if self.config.tactile:  tactile_data = np.stack(tactile_data, axis=0)   # shape is tactile=[c+p, bs, 48]

        return torch.tensor([]), torch.tensor(image_data) , torch.tensor(tactile_data)

    def build_dataset(self):
        self.total_sequences = 0
        self.sequences = []
        for episode in self.map_data:
            episode_length = episode['episode_length']
            valid_sequences = episode_length - ((self.context_len + self.prediction_horizon - 1)*self.config.sample_rate)
            if valid_sequences > 0:
                self.total_sequences += valid_sequences
                self.sequences += episode['step_save_name_list'][self.context_len + self.prediction_horizon - 1:]  #  TODO not needed for pre-loaded datasets and needs a fix for sample_rate stuff 

        self.data = []
        self.sample_index_list = []
        current_index = 0
        episode_length_list = []
        for episode in tqdm(self.map_data, desc="Loading data", dynamic_ncols=True):
            episode_length = episode['episode_length']
            episode_length_list.append(episode_length)
            for step_num, save_name in enumerate(episode['step_save_name_list']):
                if episode_length - step_num >= (self.context_len + self.prediction_horizon - 1)*self.config.sample_rate:
                    self.sample_index_list += [current_index]
                current_index += 1
                if self.config.pre_load_data:
                    step_data = np.load(save_name, allow_pickle=True)
                    image_data   = step_data[()]['image'].transpose(2, 0, 1)
                    tactile_data = step_data[()]['tactile'].transpose(2, 0, 1)
                    self.data.append([image_data, tactile_data])
                else:
                    self.data.append(save_name)
            #print the episode length on the tqdm bar
            if step_num % 10 == 0:
                tqdm.write(f"Episode length: {episode_length}")



        # log the dataset length and other features
        wandb.config.update(dict(total_sequences=self.total_sequences), allow_val_change=True)
        print("self.total_sequences: ", self.total_sequences)


def main(argv):
    ###########################
    # Setup configs
    ###########################
    config = Config()

    if "AC-VTGPT" in FLAGS.model_name:
        print("setting model to AC-VTGPT version")
        config.model_name      = "AC-VTGPT"
        config.action, config.tactile = True, True

    elif "AC-VGPT" in FLAGS.model_name:
        print("setting model to AC-VGPT version")
        config.model_name      = "AC-VGPT"
        config.action, config.tactile = True, False

    elif "VGPT" in FLAGS.model_name:
        print("setting model to VGPT version")
        config.model_name      = "VGPT"
        config.action, config.tactile = False, False

    elif "SVG-ACTP-SOP" in FLAGS.model_name:
        print("setting model to SPOTS_SVG_ACTP_SOP version")
        config.model_name      = "SVG-ACTP-SOP"
        config.action, config.tactile = True, True

    elif "SVG-ACTP" in FLAGS.model_name:
        print("setting model to SPOTS_SVG_ACTP version")
        config.model_name      = "SVG-ACTP"
        config.action, config.tactile = True, True

    elif "SVG" in FLAGS.model_name:
        print("setting model to SVG version")
        config.model_name      = "SVG"
        config.action, config.tactile = True, False

    if FLAGS.pretrained:         config.pretrained_model_path, config.pretrained_config_path                         = FLAGS.pretrained_model_path, FLAGS.pretrained_config_path
    if FLAGS.pretrained_enc:     config.load_pretrained_image_model, config.freeze_image_model                       = FLAGS.pretrained_enc, FLAGS.pretrained_enc_frozen
    if FLAGS.pretrained_ac_enc:  config.load_pretrained_ac_image_model, config.freeze_ac_image_model, config.action  = FLAGS.pretrained_ac_enc, FLAGS.pretrained_ac_enc_frozen, True
    if FLAGS.pretrained_tok:     config.load_pretrained_image_tokenizer, config.freeze_image_tokenizer               = FLAGS.pretrained_tok, FLAGS.pretrained_tok_frozen
    if FLAGS.pretrained_dec:     config.load_pretrained_image_decoder, config.freeze_image_decoder                   = FLAGS.pretrained_dec, FLAGS.pretrained_dec_frozen

    if FLAGS.num_steps  != 0:   config.num_steps = FLAGS.num_steps
    if FLAGS.model_type != '':  config.model_type = FLAGS.model_type
    if FLAGS.cluster:          config.cluster = True

    if FLAGS.train_infill:    config.train_infill   = FLAGS.train_infill
    if FLAGS.test_infill:     config.test_infill    = FLAGS.test_infill
    if config.train_infill: config.model_name += "-trn inf"
    if config.test_infill:  config.model_name += "-tst inf"

    if FLAGS.train_tactile_infill:    config.train_tactile_infill   = FLAGS.train_tactile_infill
    if FLAGS.test_tactile_infill:     config.test_tactile_infill    = FLAGS.test_tactile_infill
    if config.train_tactile_infill: config.model_name += "-trn tac inf"
    if config.test_tactile_infill:  config.model_name += "-tst tac inf"

    if FLAGS.use_all_tactile_samples: 
        config.use_all_tactile_samples = True
        config.tactile_dim = config.tactile_dim * config.sample_rate
        config.tactile_size = config.tactile_dim
        assert config.tactile_dim % config.patches_per_tactile_frame == 0

    config.test_version    = FLAGS.test_version
    config.experiment_name = FLAGS.test_version + " - " + config.model_name
    if FLAGS.model_type == "transformer":   config.model_config = model_config_builder_transformer(config)
    elif FLAGS.model_type == "SVG":         config.model_config = model_config_builder_svg(config)
    elif FLAGS.model_type == "ACTP":        config.model_config = model_config_builder_actp(config)

    ###########################
    # Setup WandB
    ###########################
    wandb.login()

    name = train_utils.format_name_with_config(config.experiment_name, config.to_dict() )
    wandb_id = "{name}_{time}".format( name=config.experiment_name, time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    wandb.init( config=config.to_dict(), id=wandb_id, name=config.experiment_name, mode="disabled" if config.debug else None, **config.wandb)

    # set run directory to save the logs    
    logging.info("Wandb logs saved to %s", wandb.run.dir)
    logging.info("Wandb url: %s", wandb.run.get_url())

    ###########################
    # Load the dataset  | load the tfrecords RLDS dataset saved locally at: /home/wmandil/tensorflow_datasets/robot_pushing_dataset/1.0.0
    ###########################
    train_dataset = VisionTactileDataset(config=config, map_file=config.dataset_train_dir + "map.npy", context_len=config.context_length,  prediction_horizon=config.num_frames - config.context_length, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    val_dataset = VisionTactileDataset(config=config, map_file=config.dataset_train_dir + "map.npy", context_len=config.context_length, prediction_horizon=config.num_frames - config.context_length, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    viz_dataset = VisionTactileDataset(config=config, map_file=config.dataset_train_dir + "map.npy", context_len=config.context_length, prediction_horizon=config.prediction_horizon, train=False, vis=True)
    viz_dataloader = DataLoader(viz_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    test_dataset = VisionTactileDataset(config=config, map_file=config.dataset_train_dir + "map.npy", context_len=config.context_length, prediction_horizon=config.prediction_horizon, train=False, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    ###########################
    # Load the model and optimizer
    ###########################
    if FLAGS.model_type == "transformer":
        model = VPGPT(config.model_config).to(config.device)
        scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device)
        plot = model.get_attention_mask()  # returns a matplotlib plot of the attention mask

    if FLAGS.model_type == "SVG":
        if config.model_name == "SVG":             model = SVG(config.model_config).to(config.device)
        elif config.model_name == "SVG-ACTP":      model = SPOTS_SVG_ACTP(config.model_config).to(config.device)
        elif config.model_name == "SVG-ACTP-SOP":  model = SPOTS_SVG_ACTP_SOP(config.model_config).to(config.device)
        model.initialise_model()

    if   config.criterion == "MAE":  criterion = nn.L1Loss()
    elif config.criterion == "MSE":  criterion = nn.MSELoss()

    ###########################
    # Save all metadata to wandb
    ###########################
    save_dir = os.path.join( config.save_dir, config.wandb["project"], wandb_id)
    wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
    logging.info("Saving to %s", save_dir)

    example_batch = next(iter(train_dataloader))
    example_batch_spec = {k: v.shape for k, v in zip(["image", "tactile"], example_batch)}
    wandb.config.update(dict(example_batch_spec=example_batch_spec), allow_val_change=True)

    # save the dataset sizes
    wandb.config.update(dict(train_dataset_size=len(train_dataset), val_dataset_size=len(val_dataset), viz_dataset_size=len(viz_dataset), test_dataset_size=len(test_dataset)), allow_val_change=True)

    # save the model attention mask
    if FLAGS.model_type == "transformer": wandb.log({"attention_mask": wandb.Image(plot)}, step=0)

    ###########################
    # Define the training functions
    ###########################
    def train_step_transformer(batch, config, model, criterion, timer):
        pred_image, image_predict, pred_tactile, tactile_predict, total_loss, loss, tactile_loss, image_context, tactile_context = train_utils.format_and_run_batch(batch, config, model, criterion, timer, horizon_rollout=False)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update_info = {"grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                       "lr": optimizer.param_groups[0]["lr"],
                       "loss": total_loss.item()}
        if config.image:    update_info["Training: image loss"]   = loss.item()
        if config.tactile:  update_info["Training: tactile loss"] = tactile_loss.item()    
        return update_info

    def train_step_svg(batch, config, model, criterion, timer):
        pred_image, image_predict, pred_tactile, tactile_predict, total_loss, loss, tactile_loss, image_context, tactile_context = train_utils.format_and_run_batch(batch, config, model, criterion, timer, horizon_rollout=False)
        update_info = {"grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                       "lr": model.prior_optimizer.param_groups[0]['lr'],
                       "loss": total_loss.item()}
        if config.image:    update_info["Training: image loss"]     = total_loss.item()
        if config.image:    update_info["Training: image kld loss"] = loss.item()
        if config.tactile:  update_info["Training: tactile loss"]   = tactile_loss.item()    
        return update_info

    def val_step(step, config, model, criterion, val_dataloader, timer):
        val_metrics = {}
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_dataloader):
                pred_image, image_predict, pred_tactile, tactile_predict, total_loss, loss, tactile_loss, image_context, tactile_context = train_utils.format_and_run_batch(batch, config, model, criterion, timer, horizon_rollout=False, repeatable_infill=True, eval=True)
                val_metrics["validation_loss"] = val_metrics.get("loss", 0) + total_loss.item()
                if config.tactile:  val_metrics["Validation: tactile loss"] = val_metrics.get("tactile_loss", 0) + tactile_loss.item()
                if config.image:    val_metrics["Validation: image loss"] = val_metrics.get("image_loss", 0) + loss.item()
        val_metrics = train_utils.tree_map(lambda x: x / (step + 1), val_metrics)
        return val_metrics

    def viz_step(step, config, model, criterion, viz_dataloader, timer):
        with torch.no_grad():
            model.eval()
            combined_losses, image_loss_list, tactile_loss_list = [], [], []
            loss_sequences_image, loss_sequences_tactile, loss_sequences_combined = [], [], []
            for i, batch in enumerate(viz_dataloader):
                if i not in config.viz_steps:  continue
                (rollout_image_prediction, image_groundtruth, rollout_tactile_prediction, tactile_groundtruth, image_losses, tactile_losses, combined_total_loss, 
                loss_sequence_image, loss_sequence_tactile, loss_sequence_combined, image_context, tactile_context) = train_utils.format_and_run_batch(batch, config, model, criterion, timer, horizon_rollout=True, repeatable_infill=True)
                if config.image:
                    # train_utils.viz_image_figure(image_groundtruth, rollout_image_prediction, image_context, config, step, step_name=i)
                    image_loss_list.append(image_losses.item())
                    loss_sequences_image.append(loss_sequence_image)
                    combined_losses.append(combined_total_loss.item())
                    loss_sequences_combined.append(loss_sequence_combined)
                if config.tactile:
                    # train_utils.viz_tactile_figure(tactile_groundtruth, rollout_tactile_prediction, tactile_context, config, step, step_name=i)
                    tactile_loss_list.append(tactile_losses.item())
                    loss_sequences_tactile.append(loss_sequence_tactile)
                if config.image and config.tactile:
                    train_utils.viz_combined_figure(image_groundtruth, rollout_image_prediction, image_context, tactile_groundtruth, rollout_tactile_prediction, tactile_context, config, step, step_name=i)

        viz_metrics = {}
        if config.model_type == "transformer":
            if config.image and config.tactile:  viz_metrics["Test: Rollout combined loss {}".format(config.prediction_horizon)] = np.mean(combined_losses)
            if config.image:                     viz_metrics["Test: Rollout image loss {}".format(config.prediction_horizon)]    = np.mean(image_loss_list)
            if config.tactile:                   viz_metrics["Test: Rollout tactile loss {}".format(config.prediction_horizon)]  = np.mean(tactile_loss_list)
        elif config.model_type == "SVG":
            if config.image and config.tactile:  viz_metrics["Test: Rollout combined loss {}".format(config.prediction_horizon)]     = np.mean(combined_losses) + np.mean(tactile_loss_list)
            if config.image:                     viz_metrics["Test: Rollout image loss {}".format(config.prediction_horizon)]        = np.mean(combined_losses)
            if config.image:                     viz_metrics["Test: Rollout image prior loss {}".format(config.prediction_horizon)]  = np.mean(image_loss_list)
            if config.tactile:                   viz_metrics["Test: Rollout tactile loss {}".format(config.prediction_horizon)]      = np.mean(tactile_loss_list)

        train_utils.viz_rollout_losses(loss_sequences_combined, loss_sequences_image, loss_sequences_tactile, config, step)
        return viz_metrics

    ###########################
    # training loop
    ###########################
    logging.info(f" --- starting training loop")

    if FLAGS.model_type == "transformer": train_step = train_step_transformer
    if FLAGS.model_type == "SVG":
        train_step = train_step_svg
        model.criterion = criterion

    model.train()
    timer = train_utils.Timer()
    step = 0
    with tqdm(total=config.num_steps, dynamic_ncols=True) as pbar:
        timer.tick("batch_gen")
        timer.tick("total")

        while step < config.num_steps:
            for batch in train_dataloader:
                timer.tock("batch_gen")

                with timer("format and train"):   update_info = train_step(batch, config, model, criterion, timer)
                timer.tock("total")

                if (step + 1) % config.log_interval == 0:
                    train_utils.wandb_log({"training": update_info, "timer": timer.get_average_times()}, step=step)

                if (step + 1) % config.eval_interval == 0:
                    with timer("val"):        
                        val_update_info = val_step(step, config, model, criterion, val_dataloader, timer)
                        train_utils.wandb_log(val_update_info, step=step)
                    with timer("visualize"):
                        viz_update_info = viz_step(step, config, model, criterion, viz_dataloader, timer)
                        train_utils.wandb_log(viz_update_info, step=step)
                    with timer("test"):  
                        viz_update_info = viz_step(step, config, model, criterion, test_dataloader, timer)
                        train_utils.wandb_log(viz_update_info, step=step)
                    model.train()

                if (step + 1) % config.save_interval == 0:
                    train_utils.save_model(model, "model_step_{i}".format(i=step), config, wandb_id)

                step += 1
                pbar.update(1)
                timer.tick("batch_gen")
                timer.tick("total")

    train_utils.save_model(model, "model_final", config, wandb_id)

if __name__ == "__main__":
    app.run(main)
