import datetime

class model_config_builder_transformer():
    def __init__(self, config):
        if config.image == True and config.action == False and config.tactile == False:
            self.block_size = int(((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width)) * config.context_length)
        if config.image == True and config.action == True and config.tactile == False:
            self.block_size = int(((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width)) * config.context_length) + ((config.context_length + 1)* config.patches_per_action_frame)
        if config.image == True and config.action == True and config.tactile == True:
            self.block_size = int(((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width)) * config.context_length) + ((config.context_length + 1)* config.patches_per_action_frame) + int((config.tactile_height / config.transformer_input_tactile_height) * (config.tactile_width / config.transformer_input_tactile_width))
        if config.image == True and config.action == False and config.tactile == True:
            self.block_size = int(((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width)) * config.context_length) + int(((config.tactile_height / config.transformer_input_tactile_height) * (config.tactile_width / config.transformer_input_tactile_width)) * config.context_length)

        self.n_layer = config.num_encoder_layers
        self.n_head = config.num_heads
        self.n_embd = config.enc_dim
        self.dropout = config.dropout
        self.input_dim = config.input_dim
        self.bias = config.bias
        
        self.H = config.image_height
        self.W = config.image_width
        self.TH = config.tactile_height
        self.TW = config.tactile_width
        self.patches_per_frame         = int((config.image_height / config.transformer_input_height) * (config.image_width / config.transformer_input_width))
        self.patches_per_frame_tactile = int((config.tactile_height / config.transformer_input_tactile_height) * (config.tactile_width / config.transformer_input_tactile_width))

        self.fh = config.transformer_input_height
        self.fw = config.transformer_input_width
        self.tfh = config.transformer_input_tactile_height
        self.tfw = config.transformer_input_tactile_width

        self.mask = config.mask
        self.num_frames = config.num_frames
        self.context_length = config.context_length
        self.prediction_length = config.num_frames - config.context_length

        self.device = config.device
        self.BeIT = config.BeIT
        self.tactile_dim = config.tactile_dim
        self.action = config.action
        self.action_dim = config.action_dim
        self.tactile = config.tactile
        self.image = config.image
        self.tactile_conditioned = config.tactile_conditioned
        self.pretrained_model_path           = config.pretrained_model_path
        self.pretrained_config_path          = config.pretrained_config_path
        self.load_pretrained_image_model     = config.load_pretrained_image_model
        self.freeze_image_model              = config.freeze_image_model
        self.load_pretrained_ac_image_model  = config.load_pretrained_ac_image_model
        self.freeze_ac_image_model           = config.freeze_ac_image_model
        self.load_pretrained_image_tokenizer = config.load_pretrained_image_tokenizer
        self.freeze_image_tokenizer          = config.freeze_image_tokenizer
        self.load_pretrained_image_decoder   = config.load_pretrained_image_decoder
        self.freeze_image_decoder            = config.freeze_image_decoder
        self.patches_per_tactile_frame       = config.patches_per_tactile_frame
        self.patches_per_action_frame        = config.patches_per_action_frame

class model_config_builder_svg():
    def __init__(self, config):
        self.lr                          = config.learning_rate
        self.beta                        = config.beta
        self.beta1                       = config.beta1
        self.batch_size                  = config.batch_size
        self.optimizer                   = config.optimizer
        self.criterion                   = config.criterion
        self.device                      = config.device
        self.g_dim                       = config.g_dim
        self.z_dim                       = config.z_dim
        self.state_action_size           = config.state_action_size
        self.rnn_size                    = config.rnn_size
        self.predictor_rnn_layers        = config.predictor_rnn_layers
        self.posterior_rnn_layers        = config.posterior_rnn_layers
        self.prior_rnn_layers            = config.prior_rnn_layers
        self.channels                    = config.channels
        self.model_dir                   = config.model_dir
        self.model_name                  = config.model_name
        self.model_name_save_appendix    = config.model_name_save_appendix
        self.n_past                      = config.n_past
        self.n_future                    = config.n_future
        self.n_eval                      = config.n_eval
        self.tactile_size                = config.tactile_size


class model_config_builder_actp():
    def __init__(self, config):
        pass


class Config:
    # add model config
    def __init__(self):
        ###########################
        # General parameters
        ###########################
        self.debug             = False
        self.pre_load_data     = True
        self.preload_data_gpu  = False
        self.model_type        = ""

        self.dataset_to_use    = "VisGel_unseen"  # robot_pushing, robot_pushing_edge_case, VisGel
        self.dataset_train_dir = "/media/wmandil/Data/Robotics/Data_sets/VisGel/data/data_unseen/images/formatted_dataset/"
        self.save_dir          = "/home/wmandil/robotics/saved_models/"

        self.model_name      = "VTGPT-infill"  # VPGPT, AC-VGPT, AC-VTGPT
        self.test_version    = "v0"
        self.experiment_name = self.test_version + " - " + self.model_name
        self.date_and_time   = datetime.datetime.now().strftime("%m%d_%H%M%S")

        self.wandb             = dict(project="SPOTS-VisGel")
        self.wandb_resume      = False
        self.wandb_resume_id   = ""

        ###########################
        # Training parameters
        ###########################
        self.seed       = 42
        self.batch_size = 128

        self.num_steps       = 50_000          # dataset is currently 144,495 steps at 256 batch size is:  560ish steps per epoch
        self.save_interval   = 10_000
        self.log_interval    = 100
        if self.debug: self.eval_interval   = 10
        else:          self.eval_interval   = 500

        self.sample_rate = 2                  # how many frames to skip for the dataset (basically makes bigger changes in between each sequence) 

        self.num_frames 	      = 5+1     # IF transformers: just context length + 1 ( + 1 because its the prediction horizon for autoregressive models)
        self.context_length       = 5
        self.prediction_horizon   = 15    # when rolling out autoregressive models, this is the prediction horizon for testing (not training)

        self.num_workers = 2
        self.device = "cuda:0"

        self.scale_data              = True

        self.use_all_tactile_samples = False

        self.blind_image_data        = False
        self.BeIT                    = False

        self.shuffle_buffer_size     = 1000
        self.val_shuffle_buffer_size = 1000

        if self.dataset_to_use == "robot_pushing":             self.viz_steps = [1, 200, 800, 1050, 1350]      # Great steps @ sample rate 10: 1 (downwards push), 1050 (upwards push), 200 (no object movement), 800 (downwards push) 1350 (upwards push)
        elif self.dataset_to_use == "robot_pushing_edge_case": self.viz_steps = [0,3,6,9, 12,15,18,21, 24,27,30,33, 36,39,42,45]  # Great steps @ sample rate 10: 1 (downwards push), 1050 (upwards push), 200 (no object movement), 800 (downwards push) 1350 (upwards push)
        else: self.viz_steps = [i for i in range(0, 1500, 50)]

        ###########################
        # Infilling parameters
        ###########################
        self.train_infill         = False
        self.test_infill          = False
        self.min_infill_patch_size  = 1
        self.max_infill_patch_size  = 64

        self.repeatable_infil_y_pos      = 15
        self.repeatable_infil_x_pos      = 15
        self.repeatable_infil_patch_size = 25

        self.train_tactile_infill = False
        self.test_tactile_infill  = False
        self.min_infill_taxels  = 1
        self.max_infill_taxels  = 64  # max 48?

        ###########################
        # optimizer parameters
        ###########################
        self.criterion = "MSE"

        self.beta1 	       = 0.9
        self.beta2 	       = 0.99
        self.weight_decay  = 1e-4 
        self.learning_rate = 0.001

        ###########################
        # SPOTS-Transformer parameters
        ###########################
        self.dtype 		            = 'float16'
        self.image 					= True
        self.action 			    = False
        self.tactile 				= True
        self.mask 			   		= True
        self.padding 				= False
        self.tactile_conditioned 	= False

        self.image_height              = 64
        self.image_width               = 64
        self.patch_size                = 16
        self.transformer_input_height  = 16
        self.transformer_input_width   = 16

        self.tactile_height                   = 64
        self.tactile_width                    = 64
        self.patch_size_tactile               = 16
        self.transformer_input_tactile_height = 16   ## TODO could be wrong here
        self.transformer_input_tactile_width  = 16   ## TODO could be wrong here


        # self.tactile_height                   = 256
        # self.tactile_width                    = 256
        # self.patch_size_tactile               = 32
        # self.transformer_input_tactile_height = 32   ## TODO could be wrong here
        # self.transformer_input_tactile_width  = 32   ## TODO could be wrong here

        self.input_dim   	           = 3
        self.action_dim 	           = 6
        self.tactile_dim 	           = 3
        self.patches_per_tactile_frame = 16  # 1 means no patches, 
        self.patches_per_action_frame  = 6   # 1 means no patches, 

        assert self.action_dim   % self.patches_per_action_frame  == 0
        assert self.image_height % self.transformer_input_height  == 0
        assert self.image_width  % self.transformer_input_width   == 0
        assert self.tactile_height % self.transformer_input_tactile_height  == 0
        assert self.tactile_width  % self.transformer_input_tactile_width   == 0

        self.enc_dim 	  	    = 768
        self.num_heads 	  	    = 12
        self.num_encoder_layers = 2
        self.dropout 		    = 0.2
        self.bias               = True   # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

        ###########################
        # Pre-loading parameters (whether to aspects of the model from a pretrained model)
        ###########################
        self.pretrained_model_path      = ""
        self.pretrained_config_path     = ""
        self.load_pretrained_image_model     = False
        self.freeze_image_model              = False

        self.load_pretrained_ac_image_model  = False
        self.freeze_ac_image_model           = False

        self.load_pretrained_image_tokenizer = False
        self.freeze_image_tokenizer          = False

        self.load_pretrained_image_decoder   = False
        self.freeze_image_decoder            = False

        ###########################
        # SPOTS-SVG parameters
        ###########################
        self.beta                      = 0.0001
        self.optimizer                 = 'adam'

        self.n_past                    = self.context_length
        self.n_future                  = self.num_frames - self.context_length
        self.n_eval                    = self.prediction_horizon
        
        self.channels                  = self.input_dim
        self.out_channels              = self.input_dim
        self.tactile_channels          = self.tactile_dim
        self.tactile_out_channels      = self.tactile_dim
        self.model_dir                 = ""
        self.model_name_save_appendix  = ""

        self.state_action_size         = self.action_dim * 2
        self.rnn_size                  = 256 # Large: 512 , Medium: 256
        self.predictor_rnn_layers      = 6   # Large: 6   , Medium: 4
        self.posterior_rnn_layers      = 4   # Large: 4   , Medium: 3
        self.prior_rnn_layers          = 4   # Large: 4   , Medium: 3
        self.g_dim                     = 256 # Large: 512 , Medium: 256
        self.z_dim                     = 10

        ###########################
        # build sub-configs
        ###########################
        self.model_config = model_config_builder_transformer(self)
        # self.model_config = model_config_builder_svg(self)

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