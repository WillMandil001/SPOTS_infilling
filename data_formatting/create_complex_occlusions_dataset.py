import os
import csv
from PIL import Image

# Define file paths
image_meta_path = "/media/wmandil/Data/Robotics/Data_sets/open_images/image_ids_and_rotation.csv"
seg_meta_path = "/media/wmandil/Data/Robotics/Data_sets/open_images/test-annotations-object-segmentation.csv"

# Define paths
image_folder = "/media/wmandil/Data/Robotics/Data_sets/open_images/test/"
segmentation_folder = "/media/wmandil/Data/Robotics/Data_sets/open_images/segmentation/"
output_pairs = []

# desired_labels = {"/m/01yrx", "/m/0bt9lr"}  # Cat and Dog label IDs

desired_labels = {
    # "/m/01mzpv",  # Chair
    "/m/02crq6",  # Couch / Sofa
    "/m/03s_tn",  # Bookshelf
    # "/m/0d4v4",   # Bed
    # "/m/03q5t",   # Dining Table
    "/m/052sf",   # Cabinet
    # "/m/018vs",   # Clipboard
    "/m/01pns0",  # Filing Cabinet
    "/m/01b9xk",  # Box
    # "/m/04dr76w", # Storage Rack
}

# desired_labels = {
#     "/m/06nrc",   # Lamp
#     "/m/02cnsz",  # Vase
#     "/m/01c648",  # Monitor
#     "/m/01m2v",   # Laptop
#     "/m/03qjg",   # Houseplant
# }

# desired_labels = {
#     "/m/014j1m",  # Apple
#     "/m/09qck",   # Banana
#     "/m/0fm3zh",  # Cucumber
#     "/m/07j87",   # Tomato
#     "/m/09k_b",   # Carrot
#     "/m/02xwb",   # Lemon
#     "/m/06k2mb",  # Orange
#     "/m/07cx4",   # Nuts
#     "/m/01b3n",   # Egg
#     "/m/02cvgx",  # Garlic
#     "/m/01f91_",  # Pepper
#     "/m/05zsy",   # Potato
#     "/m/0l515",   # Milk Carton/Bottle
#     "/m/09728",   # Bread
#     "/m/01g317",  # Burger
#     "/m/0663v",   # Pizza
#     "/m/01dwwc",  # Hotdog
#     "/m/021mn",   # Cookies
#     "/m/0fszt",   # Juice Box
#     "/m/01fqt",   # Chocolate
#     "/m/07clx",   # Doughnut
#     "/m/02p5f1q", # Soft Drink Bottle
#     "/m/02jvh9",  # Tea Cup / Coffee Cup
#     "/m/01pns0",  # Wine Bottle
#     "/m/03y6mg",  # Honey Jar
#     "/m/04ctx",   # Plate
#     "/m/0l515",   # Bowl
#     "/m/0dt3t",   # Cutlery (Fork/Spoon/Knife)
#     "/m/04h7h",   # Cutting Board
#     "/m/04szw",   # Jar / Container
# }

desired_labels = {
    "/m/03m3pdh" # Couch / Sofa
    "/m/0h8k0z3"
}

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
        label = row[2].strip()          # LabelName (e.g., /m/01yrx for cat)

        # if image_id in test_image_ids and label in desired_labels:
        image_path = os.path.join(image_folder, image_id + ".jpg")
        mask_path = os.path.join(segmentation_folder, mask_filename)
        output_pairs.append((image_path, mask_path, label))

# Print first few matches to confirm
for img, mask, label in output_pairs[:10]:
    print(f"Image: {img} -> Mask: {mask}")

# Define save location
dataset_save_location = "/media/wmandil/Data/Robotics/Data_sets/open_images_mask_dataset/"
os.makedirs(dataset_save_location, exist_ok=True)

# Process each image-mask pair
for img_path, mask_path, label in output_pairs[:5000]:
    try:
        # Open the image (RGB) and mask (grayscale)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Resize mask to match image size if they are different
        if mask.size != image.size:
            print(f"Resizing mask {mask_path} from {mask.size} to {image.size}")
            mask = mask.resize(image.size, Image.NEAREST)  # Use NEAREST to preserve segmentation edges

        # Create an RGBA image (adds transparency where mask is black)
        image_rgba = image.copy()
        image_rgba.putalpha(mask)  # Use mask as alpha channel (transparency)

        # crop image to be around the mask only and not the whole image
        bbox = mask.getbbox()
        image_rgba = image_rgba.crop(bbox)

        # Resize the image to 256x256  # module 'PIL.Image' has no attribute 'ANTIALIAS'
        image_rgba = image_rgba.resize((256, 256), Image.Resampling.LANCZOS)

        # Save the masked object as PNG (preserving transparency)
        # remove / from the label
        label = label.replace("/", "")
        print (label)
        # create dataset_save_location + label
        os.makedirs(dataset_save_location + label, exist_ok=True)
        save_path = os.path.join(dataset_save_location + label, os.path.basename(img_path).replace(".jpg", ".png"))
        image_rgba.save(save_path, format="PNG")

        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Error processing {img_path}, {mask_path}: {e}")