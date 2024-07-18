#!/bin/bash

python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VGPT"  --test_version="v4" --infill=False
python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT" --test_version="v4" --infill=False
python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="VGPT"     --test_version="v4" --infill=False

python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VGPT"  --test_version="v4" --infill=True
python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="AC-VTGPT" --test_version="v4" --infill=True
python /home/wmandil/robotics/SPOTS_infilling/train.py --model_name="VGPT"     --test_version="v4" --infill=True
