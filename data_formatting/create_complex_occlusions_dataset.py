import random
import numpy as np
from PIL import Image

def apply_random_occlusion(
    background,
    occluder,
    mask,
    scale_min=0.2,
    scale_max=0.5
):
    """
    Places an occluding object (e.g., a fridge) on a background image (e.g., a robot scene),
    using the provided segmentation mask. The occluder will be upright,
    but randomly scaled and positioned.
    """

    # 1. Load the images and convert to RGBA
    background = background.convert("RGBA")
    occluder = occluder.convert("RGBA")
    mask = mask.convert("L")  # 'L' = 8-bit pixels, black and white

    # 2. Randomly scale the occluder
    scale_factor = random.uniform(scale_min, scale_max)
    new_width = int(occluder.width * scale_factor)
    new_height = int(occluder.height * scale_factor)

    occluder = occluder.resize((new_width, new_height), resample=Image.BICUBIC)
    mask = mask.resize((new_width, new_height), resample=Image.BICUBIC)

    # 3. Randomly pick a top-left coordinate for the occluder
    #    Ensuring it is fully within the background (or at least not out-of-bounds)
    max_x = max(1, background.width - new_width)
    max_y = max(1, background.height - new_height)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # 4. Create a temporary layer for placing the occluder
    occluder_layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    occluder_layer.paste(occluder, (x, y), mask=mask)

    # 5. Composite the occluder layer onto the background
    out = Image.alpha_composite(background, occluder_layer)

    return out


# import csv

# # Define file paths
# image_meta_path = "/media/wmandil/Data/Robotics/Data_sets/open_images/image_ids_and_rotation.csv"
# seg_meta_path = "/media/wmandil/Data/Robotics/Data_sets/open_images/test-annotations-object-segmentation.csv"

# # Load test image IDs
# test_image_ids = set()
# with open(image_meta_path, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)  # Skip header
#     for row in reader:
#         if row[1].strip().lower() == "test":  # Ensure it's from the test set
#             test_image_ids.add(row[0].strip())

# # Load segmentation image IDs
# seg_image_ids = set()
# with open(seg_meta_path, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)  # Skip header
#     for row in reader:
#         seg_image_ids.add(row[1].strip())  # ImageID is in column 2

# # Find common image IDs
# common_ids = test_image_ids.intersection(seg_image_ids)

# print(f"Total test images: {len(test_image_ids)}")
# print(f"Total segmented images: {len(seg_image_ids)}")
# print(f"Matching image IDs: {len(common_ids)}")

# # Print some matches
# print("Example matching IDs:", list(common_ids)[:10])


import os
import csv

# Define file paths
image_meta_path = "/media/wmandil/Data/Robotics/Data_sets/open_images/image_ids_and_rotation.csv"
seg_meta_path = "/media/wmandil/Data/Robotics/Data_sets/open_images/test-annotations-object-segmentation.csv"

# Define paths
image_folder = "/media/wmandil/Data/Robotics/Data_sets/open_images/test/"
segmentation_folder = "/media/wmandil/Data/Robotics/Data_sets/open_images/segmentation/"
output_pairs = []

# Load test image IDs
test_image_ids = set()
with open(image_meta_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if row[1].strip().lower() == "test":  # Ensure test images only
            test_image_ids.add(row[0].strip())

# Read segmentation metadata and find matching pairs
with open(seg_meta_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        image_id = row[1].strip()
        mask_filename = row[0].strip()  # MaskPath is in first column

        if image_id in test_image_ids:
            image_path = os.path.join(image_folder, image_id + ".jpg")
            mask_path = os.path.join(segmentation_folder, mask_filename)
            output_pairs.append((image_path, mask_path))

# Print first few matches to confirm
for img, mask in output_pairs[:10]:
    print(f"Image: {img} -> Mask: {mask}")