#!/bin/bash

# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT"  --model_type="transformer" --test_version="-prim-xela-timesteps"  --object_mask_infill=False --complex_shape_infill=True
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VGPT"   --model_type="transformer" --test_version="-prim-xela-timesteps"  --object_mask_infill=False --complex_shape_infill=True
python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="SVG"       --model_type="SVG"         --test_version="-prim-xela-timesteps"  --object_mask_infill=False --complex_shape_infill=True
python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="SVG-ACTP"  --model_type="SVG"         --test_version="-prim-xela-timesteps"  --object_mask_infill=False --complex_shape_infill=True

# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT" --object_mask_infill=True  --complex_shape_infill=False
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="SVG-ACTP" --object_mask_infill=True  --complex_shape_infill=False
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="SVG"      --object_mask_infill=True  --complex_shape_infill=False


# test with pre-trained tokenizer
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VGPT"  --test_version="v4 - pt-infill-tok " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_tok=True --pretrained_tok_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/VGPT-infill/v4 - VGPT-infill_20240717_082440/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_082440-v4 - VGPT-infill_20240717_082440/files/config.yaml"
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT" --test_version="v4 - pt-infill-tok " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_tok=True --pretrained_tok_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/VGPT-infill/v4 - VGPT-infill_20240717_082440/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_082440-v4 - VGPT-infill_20240717_082440/files/config.yaml"
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="VGPT"     --test_version="v4 - pt-infill-tok " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_tok=True --pretrained_tok_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/VGPT-infill/v4 - VGPT-infill_20240717_082440/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_082440-v4 - VGPT-infill_20240717_082440/files/config.yaml"

# test with pre-trained encoder
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VGPT"  --test_version="v4 - pt-infill-enc " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_enc=True --pretrained_enc_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/VGPT-infill/v4 - VGPT-infill_20240717_082440/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_082440-v4 - VGPT-infill_20240717_082440/files/config.yaml"
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT" --test_version="v4 - pt-infill-enc " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_enc=True --pretrained_enc_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/VGPT-infill/v4 - VGPT-infill_20240717_082440/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_082440-v4 - VGPT-infill_20240717_082440/files/config.yaml"
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="VGPT"     --test_version="v4 - pt-infill-enc " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_enc=True --pretrained_enc_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/VGPT-infill/v4 - VGPT-infill_20240717_082440/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_082440-v4 - VGPT-infill_20240717_082440/files/config.yaml"

# test with pre-trained ac-encoder
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VGPT"  --test_version="v4 - pt-infill-acenc " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_ac_enc=True --pretrained_ac_enc_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/AC-VGPT-infill/v4 - AC-VGPT-infill_20240717_023546/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_023546-v4 - AC-VGPT-infill_20240717_023546/files/config.yaml"
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT" --test_version="v4 - pt-infill-acenc " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_ac_enc=True --pretrained_ac_enc_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/AC-VGPT-infill/v4 - AC-VGPT-infill_20240717_023546/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_023546-v4 - AC-VGPT-infill_20240717_023546/files/config.yaml"
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="VGPT"     --test_version="v4 - pt-infill-acenc " --infill=True   --num_steps=10_000 --pretrained=True --pretrained_ac_enc=True --pretrained_ac_enc_frozen=True --pretrained_model_path="/home/wmandil/robotics/saved_models/AC-VGPT-infill/v4 - AC-VGPT-infill_20240717_023546/model_final.pth" --pretrained_config_path="/home/wmandil/robotics/SPOTS_infilling/wandb/run-20240717_023546-v4 - AC-VGPT-infill_20240717_023546/files/config.yaml"

# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VGPT"  --test_version="v4" --infill=True
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT" --test_version="v4" --infill=True
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="VGPT"     --test_version="v4" --infill=True

# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VGPT"  --test_version="v4" --infill=False
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT" --test_version="v4" --infill=False
# python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="VGPT"     --test_version="v4" --infill=False