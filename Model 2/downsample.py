import os
import numpy as np
import cv2
from tqdm import tqdm

# Define paths
training_images_path = './Training_Images/'
ground_truth_path = './Ground_Truth/'
dataset_path = './dataset/'
scales = {'twenty_five': 0.25, 'fifty': 0.5, 'original': 1.0}

# Validate input paths
if not os.path.exists(training_images_path):
    print(f"Error: Training images path '{training_images_path}' does not exist.")
    exit(1)
if not os.path.exists(ground_truth_path):
    print(f"Error: Ground truth path '{ground_truth_path}' does not exist.")
    exit(1)

# Create directory structure
for scale_name in scales:
    os.makedirs(os.path.join(dataset_path, scale_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, scale_name, 'masks'), exist_ok=True)

# Get all valid image files
image_files = [f for f in os.listdir(training_images_path) if f.endswith('.jpg')]
print(f"Found {len(image_files)} images to process")

# Validate that all images have corresponding masks
valid_files = []
for file_name in image_files:
    mask_path = os.path.join(ground_truth_path, file_name.replace('.jpg', '.png'))
    if os.path.exists(mask_path):
        valid_files.append(file_name)
    else:
        print(f"Warning: Mask not found for {file_name}, skipping...")

print(f"Processing {len(valid_files)} valid image-mask pairs")

# Process images
for file_name in tqdm(valid_files, desc="Downsampling images"):
    # Load image and mask
    image_path = os.path.join(training_images_path, file_name)
    mask_path = os.path.join(ground_truth_path, file_name.replace('.jpg', '.png'))
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {file_name}, skipping...")
        continue
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not read mask for {file_name}, skipping...")
        continue
    
    # Process each scale
    for scale_name, scale_factor in scales.items():
        # For original scale, just copy the files
        if scale_factor == 1.0:
            cv2.imwrite(os.path.join(dataset_path, scale_name, 'images', file_name), image)
            cv2.imwrite(os.path.join(dataset_path, scale_name, 'masks', file_name.replace('.jpg', '.png')), mask)
            continue
        
        # Calculate new dimensions
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Ensure valid dimensions (at least 1x1)
        if new_h < 1 or new_w < 1:
            print(f"Warning: Scaled dimensions too small for {file_name} at {scale_name} scale, skipping...")
            continue
        
        # Resize image and mask
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Save files
        cv2.imwrite(os.path.join(dataset_path, scale_name, 'images', file_name), scaled_image)
        cv2.imwrite(os.path.join(dataset_path, scale_name, 'masks', file_name.replace('.jpg', '.png')), scaled_mask)

print("Multi-scale dataset creation completed successfully!")