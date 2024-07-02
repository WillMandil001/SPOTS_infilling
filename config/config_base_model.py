import datetime

class model_config_builder():
    def __init__(self, config):
        if config.image == True and config.tactile == False and config.action == False:
            self.block_size = int(((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width)) * config.context_length)
        if config.image == True and config.action == True and config.tactile == False:
            self.block_size = int(((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width)) * config.context_length) + (config.context_length + 1)
        if config.image == True and config.action == True and config.tactile == True:
            self.block_size = int(((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width)) * config.context_length) + (config.context_length + 1) + config.context_length

        self.n_layer = config.num_encoder_layers
        self.n_head = config.num_heads
        self.n_embd = config.enc_dim
        self.dropout = config.dropout
        self.input_dim = config.input_dim
        self.H = config.image_height
        self.bias = config.bias
        self.W = config.image_width
        self.fh = config.transformer_input_height
        self.fw = config.transformer_input_width
        self.mask = config.mask
        self.num_frames = config.num_frames
        self.context_length = config.context_length
        self.prediction_length = config.num_frames - config.context_length
        self.patches_per_frame =  int((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width))
        self.device = config.device
        self.BeIT = config.BeIT
        self.tactile_dim = config.tactile_dim
        self.action = config.action
        self.action_dim = config.action_dim
        self.tactile = config.tactile
        self.image = config.image
        self.tactile_conditioned = config.tactile_conditioned
        self.pretrained_acvp_model_path = config.pretrained_acvp_model_path

class Config:
    # add model config
    def __init__(self):
        ###########################
        #
        # General parameters
        #
        ###########################
        self.debug             = False

        self.pre_load_data     = True
        self.dataset_name      = "robot_grasping_dataset"
        self.dataset_train_dir = "/home/wmandil/robotics/datasets/robot_pushing/train/formatted_dataset/"  # for the cluster machine: /shared/home/wmandil/datasets/robot_pushing/train/formatted_dataset/
        self.dataset_val_dir   = "/home/wmandil/robotics/datasets/robot_pushing/val/formatted_dataset/"    # for the cluster machine: /shared/home/wmandil/datasets/robot_pushing/val/formatted_dataset/
        self.save_dir          = "/home/wmandil/robotics/saved_models/"                                    # for the cluster machine: /shared/home/wmandil/saved_models/

        # if you moved the dataset post formatting, you can use the following to replace the old path with the new one
        self.to_replace   = "/media/wmandil/Data/Robotics/Data_sets/single_object_velocity_controlled_dataset/single_object_velocity_controlled_dataset/"
        self.replace_with = "/home/wmandil/robotics/datasets/robot_pushing/"  # for the cluster machine: /shared/home/wmandil/datasets/robot_pushing/

        self.model_name      = "ACVTPGPT"
        self.experiment_name = self.model_name
        self.date_and_time   = datetime.datetime.now().strftime("%m%d_%H%M%S")

        self.wandb             = dict(project="SPOTS_pushing_debug")
        self.wandb_resume      = False
        self.wandb_resume_id   = ""

        ###########################
        #
        # Training parameters
        #
        ###########################
        self.seed       = 42
        self.batch_size = 256

        self.num_steps       = 25_000          # dataset is currently 144,495 steps at 256 batch size is:  560ish steps per epoch
        self.eval_interval   = 10 # 500
        self.save_interval   = 1_000
        self.log_interval    = 100

        self.sample_rate = 10                  # how many frames to skip for the dataset (basically makes bigger changes in between each sequence) 

        self.num_frames 	      = 10 + 1     # just context length + 1 ( + 1 because its the prediction horizon for autoregressive models)
        self.context_length       = 10
        self.prediction_horizon   = 20         # when rolling out autoregressive models, this is the prediction horizon for testing (not training)

        self.num_workers = 4
        self.device = "cuda"

        self.load_full_dataset_to_gpu = True

        self.scale_data            = True

        self.infill_patches        = True
        self.blind_image_data      = False
        self.BeIT                  = False

        self.shuffle_buffer_size = 1000
        self.val_shuffle_buffer_size = 1000

        self.viz_steps = [1, 200, 800, 1050, 1350]  # Great steps @ sample rate 10: 1 (downwards push), 1050 (upwards push), 200 (no object movement), 800 (downwards push) 1350 (upwards push)

        ###########################
        #
        # optimizer parameters
        #
        ###########################
        self.criterion = "MAE"

        self.beta1 	       = 0.9 
        self.beta2 	       = 0.99 
        self.weight_decay  = 1e-4 
        self.learning_rate = 0.001

        ###########################
        #
        # Transformer parameters
        #
        ###########################
        self.image_height             = 64
        self.image_width              = 64
        self.patch_size               = 16
        self.transformer_input_height = 16
        self.transformer_input_width  = 16

        self.input_dim   	   = 3
        self.action_dim 	   = 6
        self.tactile_dim 	   = 48

        self.enc_dim 	  	    = 768
        self.num_heads 	  	    = 12
        self.num_encoder_layers = 2
        self.dropout 		    = 0.2
        self.bias               = True   # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

        self.dtype 		            = 'float16' 
        self.image 					= True 
        self.action 			    = True 
        self.tactile 				= True 
        self.mask 			   		= True 
        self.padding 				= False 
        self.tactile_conditioned 	= False 
        self.pretrained_acvp_model_path = ""

        self.model_config = model_config_builder(self)
    
    def to_dict(self):
        ''' returns a dictionary of all the self variables and nest the model_config as well
            the function must be repeatable '''
        def recursive_to_dict(obj):
            if isinstance(obj, dict):
                return {k: recursive_to_dict(v) for k, v in obj.items()}
            elif hasattr(obj, "__dict__"):
                return {k: recursive_to_dict(v) for k, v in obj.__dict__.items()}
            else:
                return obj
        
        return recursive_to_dict(self)

    def __repr__(self):
        return str(self.to_dict())