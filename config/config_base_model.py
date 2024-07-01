import torch.nn as nn
import datetime

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

def get_config():
    config = dict(
        pre_load_data     = True,
        dataset_name      = "robot_grasping_dataset",
        dataset_train_dir = "/home/wmandil/robotics/datasets/robot_pushing/train/formatted_dataset/",
        dataset_val_dir   = "/home/wmandil/robotics/datasets/robot_pushing/val/formatted_dataset/",
        save_dir          = "/home/wmandil/robotics/saved_models/",

        # if you moved the dataset post formatting, you can use the following to replace the old path with the new one
        to_replace = "/media/wmandil/Data/Robotics/Data_sets/single_object_velocity_controlled_dataset/single_object_velocity_controlled_dataset/",
        replace_with = "/home/wmandil/robotics/datasets/robot_pushing/",

        model_name      = "ACVTPGPT" ,
        experiment_name = "robot_pushing_test_001",
        date_and_time   = datetime.datetime.now().strftime("%m%d_%H%M%S"),

        wandb             = dict(project="SPOTS_pushing_test", group=placeholder(str), entity=placeholder(str)),
        wandb_resume      = False,
        wandb_resume_id   = "",

        ###########################
        #
        # Training parameters
        #
        ###########################
        seed = 42,
        batch_size = 256,

        num_steps       = 1_000_000,             # dataset is currently 144,495 steps long (with batch size of 128 = 1200 batchs (num_steps) per epoch. and this would be 8000 epochs)
        eval_interval   = 10, # 2_000,
        save_interval   = 20_000,
        log_interval    = 100,

        sample_rate = 10,                  # how many frames to skip for the dataset (basically makes bigger changes in between each sequence) 

        num_frames 	         = 10 + 1,     # just context length + 1 ( + 1 because its the prediction horizon for autoregressive models)
        context_length       = 10,
        prediction_horizon   = 20,         # when rolling out autoregressive models, this is the prediction horizon for testing (not training)

        num_workers = 4,
        device = "cuda",

        load_full_dataset_to_gpu = True,

        infill_patches        = True,
        scale_tactile_tactile = True,
        blind_image_data      = False,
        BeIT                  = False,

        shuffle_buffer_size = 1000,
        val_shuffle_buffer_size = 1000,

        viz_steps = [1, 200, 800, 1050, 1350],  # Great steps @ sample rate 10: 1 (downwards push), 1050 (upwards push), 200 (no object movement), 800 (downwards push) 1350 (upwards push)

        ###########################
        #
        # optimizer parameters
        #
        ###########################
        criterion = "MAE",

        beta1 	      = 0.9, 
        beta2 	      = 0.99, 
        weight_decay  = 1e-4, 
        learning_rate = 0.001,

        ###########################
        #
        # Transformer parameters
        #
        ###########################
        image_height = 64,
        image_width  = 64,
        patch_size   = 16,
        transformer_input_height = 16,
        transformer_input_width  = 16,
        input_dim   	   = 3,
        action_dim 		   = 6,
        tactile_dim 	   = 48,

        enc_dim 	  	   = 768,
        num_heads 	  	   = 12,
        num_encoder_layers = 2,
        dropout 		   = 0.2,
        bias               = True,   # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

        dtype 		            = 'float16', 
        image 					= True, 
        action 					= True, 
        tactile 				= True, 
        mask 			   		= True, 
        padding 				= False, 
        tactile_conditioned 	= False, 
        pretrained_acvp_model_path = "",
        )

    ###########################
    #
    # create specific model config for our transformer library to use (build from the config above)
    #
    ###########################
    if config["image"] == True and config["tactile"] == False and config["action"] == False:
        config["block_size"] = int(((config["image_height"] / config["transformer_input_height"]) * (config["image_width"] / config["transformer_input_width"])) * config["context_length"])
    if config["image"] == True and config["action"] == True and config["tactile"] == False:
        config["block_size"] = int(((config["image_height"] / config["transformer_input_height"]) * (config["image_width"] / config["transformer_input_width"])) * config["context_length"]) + (config["context_length"] + 1)
    if config["image"] == True and config["action"] == True and config["tactile"] == True:
        config["block_size"] = int(((config["image_height"] / config["transformer_input_height"]) * (config["image_width"] / config["transformer_input_width"])) * config["context_length"]) + (config["context_length"] + 1) + config["context_length"]

    model_config = dict(
        block_size = config["block_size"],
        n_layer = config["num_encoder_layers"],
        n_head = config["num_heads"],
        n_embd = config["enc_dim"],
        dropout = config["dropout"],
        input_dim = config["input_dim"],
        H = config["image_height"],
        bias = config["bias"],
        W = config["image_width"],
        fh = config["transformer_input_height"],
        fw = config["transformer_input_width"],
        mask = config["mask"],
        num_frames = config["num_frames"],
        context_length = config["context_length"],
        prediction_length = config["num_frames"] - config["context_length"],
        patches_per_frame =  int((config["image_height"] / config["transformer_input_height"]) * (config["image_width"] / config["transformer_input_width"])),
        device = config["device"],
        BeIT = config["BeIT"],
        tactile_dim = config["tactile_dim"],
        action = config["action"],
        action_dim = config["action_dim"],
        tactile = config["tactile"],
        image = config["image"],
        tactile_conditioned = config["tactile_conditioned"],
        pretrained_acvp_model_path = config["pretrained_acvp_model_path"],
    )

    config["model_config"] = model_config

    return ConfigDict(config)