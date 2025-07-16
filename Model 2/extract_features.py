import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from scipy.ndimage import distance_transform_edt
import random
from tqdm import tqdm

# Define paths
dataset_path = './dataset/'
csv_output_path = './csv_datasets/'
os.makedirs(csv_output_path, exist_ok=True)

def extract_pixel_features(img, x, y, mask, superpixels):
    """Extract comprehensive features for a pixel at position (x, y)"""
    h, w = img.shape[:2]
    features = {}
    
    # Validate input coordinates
    if not (0 <= x < h and 0 <= y < w):
        return None
    
    # 1. COLOR FEATURES
    # RGB values of the center pixel
    features['r'] = float(img[x, y, 0])
    features['g'] = float(img[x, y, 1])
    features['b'] = float(img[x, y, 2])
    
    # Convert to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    features['h'] = float(hsv_img[x, y, 0])
    features['s'] = float(hsv_img[x, y, 1])
    features['v'] = float(hsv_img[x, y, 2])
    
    # Convert to LAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    features['l'] = float(lab_img[x, y, 0])
    features['a'] = float(lab_img[x, y, 1])
    features['b_val'] = float(lab_img[x, y, 2])
    
    # 2. SPATIAL FEATURES
    # Normalized coordinates
    features['rel_x'] = float(x / h)
    features['rel_y'] = float(y / w)
    
    # Distance from center
    center_x, center_y = h // 2, w // 2
    dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    features['dist_to_center'] = float(dist_to_center / np.sqrt(center_x**2 + center_y**2))
    
    # Distance from borders
    features['dist_to_top'] = float(x)
    features['dist_to_left'] = float(y)
    features['dist_to_bottom'] = float(h - x - 1)
    features['dist_to_right'] = float(w - y - 1)
    features['min_dist_to_border'] = float(min(x, y, h - x - 1, w - y - 1))
    
    # 3. MULTI-SCALE LOCAL FEATURES
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    for window_size in [5, 11, 21]:  # Multiple scales
        half_w = window_size // 2
        x_min = max(0, x - half_w)
        x_max = min(h, x + half_w + 1)
        y_min = max(0, y - half_w)
        y_max = min(w, y + half_w + 1)
        
        # Ensure window has adequate size
        if x_max <= x_min + 2 or y_max <= y_min + 2:
            # Window too small, use fallback values
            for c in ['r', 'g', 'b']:
                features[f'{c}_mean_{window_size}'] = features[c]
                features[f'{c}_std_{window_size}'] = 0.0
            features[f'lbp_mean_{window_size}'] = 0.0
            features[f'gradient_mag_{window_size}'] = 0.0
            for i in range(8):
                features[f'grad_dir_hist_{i}_{window_size}'] = 0.0
            features[f'variance_{window_size}'] = 0.0
            features[f'entropy_{window_size}'] = 0.0
            continue
        
        # Extract window
        window = img[x_min:x_max, y_min:y_max]
        gray_window = gray_img[x_min:x_max, y_min:y_max]
        
        # Color statistics in window
        for c_idx, c in enumerate(['r', 'g', 'b']):
            values = window[:, :, c_idx].flatten()
            features[f'{c}_mean_{window_size}'] = float(np.mean(values))
            features[f'{c}_std_{window_size}'] = float(np.std(values))
        
        # Local Binary Pattern
        if gray_window.shape[0] > 3 and gray_window.shape[1] > 3:
            lbp = local_binary_pattern(gray_window, 8, 1, method='uniform')
            features[f'lbp_mean_{window_size}'] = float(np.mean(lbp))
            features[f'lbp_std_{window_size}'] = float(np.std(lbp))
        else:
            features[f'lbp_mean_{window_size}'] = 0.0
            features[f'lbp_std_{window_size}'] = 0.0
        
        # Gradient features
        sobel_x = cv2.Sobel(gray_window, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_window, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features[f'gradient_mag_{window_size}'] = float(np.mean(gradient_mag))
        
        # Gradient direction histogram (8 bins)
        if np.count_nonzero(gradient_mag) > 0:
            gradient_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
            hist, _ = np.histogram(gradient_dir, bins=8, range=(-180, 180))
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
            for i in range(8):
                features[f'grad_dir_hist_{i}_{window_size}'] = float(hist[i])
        else:
            for i in range(8):
                features[f'grad_dir_hist_{i}_{window_size}'] = 0.0
        
        # Texture complexity
        features[f'variance_{window_size}'] = float(np.var(gray_window))
        
        # Local entropy (measure of randomness)
        histogram, _ = np.histogram(gray_window, bins=32, range=(0, 256))
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
            # Add small epsilon to avoid log(0)
            entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
            features[f'entropy_{window_size}'] = float(entropy)
        else:
            features[f'entropy_{window_size}'] = 0.0
    
    # 4. EDGE FEATURES
    # Compute edges and distance to nearest edge
    edges = cv2.Canny(gray_img, 100, 200)
    
    # Distance transform from edges
    if np.any(edges > 0):  # Check if any edges were detected
        dist_transform = distance_transform_edt(edges == 0)
        features['dist_to_edge'] = float(dist_transform[x, y])
    else:
        # No edges detected, use fallback
        features['dist_to_edge'] = float(min(x, y, h-x-1, w-y-1))  # Distance to image border
    
    # Edge density in local neighborhood
    for radius in [5, 11, 21]:
        half_r = radius // 2
        x_min = max(0, x - half_r)
        x_max = min(h, x + half_r + 1)
        y_min = max(0, y - half_r)
        y_max = min(w, y + half_r + 1)
        
        if x_max > x_min and y_max > y_min:
            local_edges = edges[x_min:x_max, y_min:y_max]
            edge_density = np.sum(local_edges > 0) / (local_edges.shape[0] * local_edges.shape[1])
            features[f'edge_density_{radius}'] = float(edge_density)
        else:
            features[f'edge_density_{radius}'] = 0.0
    
    # 5. SUPERPIXEL FEATURES
    if superpixels is not None:
        # Get superpixel ID
        sp_id = superpixels[x, y]
        features['sp_id'] = int(sp_id)
        
        # Get all pixels in this superpixel
        sp_mask = (superpixels == sp_id)
        sp_coords = np.where(sp_mask)
        
        if len(sp_coords[0]) > 0:
            # Superpixel center coordinates
            sp_center_x = np.mean(sp_coords[0])
            sp_center_y = np.mean(sp_coords[1])
            
            # Distance to superpixel center
            dist_to_sp_center = np.sqrt((x - sp_center_x)**2 + (y - sp_center_y)**2)
            features['dist_to_sp_center'] = float(dist_to_sp_center)
            
            # Superpixel size
            sp_size = len(sp_coords[0])
            features['sp_size'] = float(sp_size)
            features['sp_size_rel'] = float(sp_size / (h * w))
            
            # Superpixel color features
            sp_pixels_r = img[:, :, 0][sp_mask]
            sp_pixels_g = img[:, :, 1][sp_mask]
            sp_pixels_b = img[:, :, 2][sp_mask]
            
            features['sp_r_mean'] = float(np.mean(sp_pixels_r))
            features['sp_g_mean'] = float(np.mean(sp_pixels_g))
            features['sp_b_mean'] = float(np.mean(sp_pixels_b))
            features['sp_r_std'] = float(np.std(sp_pixels_r))
            features['sp_g_std'] = float(np.std(sp_pixels_g))
            features['sp_b_std'] = float(np.std(sp_pixels_b))
            
            # Contrast between pixel and superpixel mean
            features['contrast_to_sp_r'] = float(abs(features['r'] - features['sp_r_mean']))
            features['contrast_to_sp_g'] = float(abs(features['g'] - features['sp_g_mean']))
            features['contrast_to_sp_b'] = float(abs(features['b'] - features['sp_b_mean']))
        else:
            # Fallback for empty superpixel (shouldn't happen, but just in case)
            features['dist_to_sp_center'] = 0.0
            features['sp_size'] = 0.0
            features['sp_size_rel'] = 0.0
            features['sp_r_mean'] = features['r']
            features['sp_g_mean'] = features['g']
            features['sp_b_mean'] = features['b']
            features['sp_r_std'] = 0.0
            features['sp_g_std'] = 0.0
            features['sp_b_std'] = 0.0
            features['contrast_to_sp_r'] = 0.0
            features['contrast_to_sp_g'] = 0.0
            features['contrast_to_sp_b'] = 0.0
    
    return features

def create_scale_dataset(scale_name, samples_per_class=100):
    """Create dataset for a specific scale with balanced sampling"""
    print(f"Creating dataset for {scale_name} scale...")
    
    # Get image file list
    image_dir = os.path.join(dataset_path, scale_name, 'images')
    mask_dir = os.path.join(dataset_path, scale_name, 'masks')
    
    # Validate directories
    if not os.path.exists(image_dir):
        print(f"Error: Image directory for {scale_name} scale not found.")
        return
    if not os.path.exists(mask_dir):
        print(f"Error: Mask directory for {scale_name} scale not found.")
        return
    
    # Get list of valid image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    valid_files = []
    
    for img_file in image_files:
        mask_file = img_file.replace('.jpg', '.png')
        if os.path.exists(os.path.join(mask_dir, mask_file)):
            valid_files.append(img_file)
        else:
            print(f"Warning: Mask not found for {img_file}, skipping...")
    
    if not valid_files:
        print(f"No valid image-mask pairs found for {scale_name} scale.")
        return
    
    print(f"Found {len(valid_files)} valid image-mask pairs for {scale_name} scale.")
    
    # Define placeholder columns for lower resolution predictions
    placeholders = []
    if scale_name == 'fifty':
        placeholders = ['twenty_five_pred']
    elif scale_name == 'original':
        placeholders = ['twenty_five_pred', 'fifty_pred']
    
    # Process in batches to manage memory
    dataset_rows = []
    batch_size = 100  # Process this many images before writing to CSV
    
    # Create output file and write header first
    output_file = os.path.join(csv_output_path, f"{scale_name}_features.csv")
    
    for batch_idx in range(0, len(valid_files), batch_size):
        batch_files = valid_files[batch_idx:batch_idx+batch_size]
        
        for img_file in tqdm(batch_files, desc=f"Processing batch {batch_idx//batch_size + 1}/{(len(valid_files)-1)//batch_size + 1}"):
            # Load image and mask
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file.replace('.jpg', '.png'))
            
            # Read image and mask
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_file}, skipping...")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask for {img_file}, skipping...")
                continue
            
            # Ensure mask is binary (0 or 1)
            mask = (mask > 127).astype(np.uint8)
            
            # Compute superpixels for region-based features
            h, w = img.shape[:2]
            # Adjust number of segments based on image size
            n_segments = max(50, min(500, (h * w) // 1000))
            
            # Compute superpixels (with fallback if SLIC fails)
            try:
                superpixels = slic(img, n_segments=n_segments, compactness=10, sigma=1, start_label=1)
            except:
                # Fallback: create a simple grid
                grid_size = max(10, min(50, n_segments // 10))
                y_grid, x_grid = np.meshgrid(
                    np.linspace(0, grid_size-1, w, dtype=int),
                    np.linspace(0, grid_size-1, h, dtype=int)
                )
                superpixels = x_grid * grid_size + y_grid
            
            # Find foreground and background pixels
            fg_coords = np.where(mask > 0)
            bg_coords = np.where(mask == 0)
            
            # Skip if no foreground or background pixels
            if len(fg_coords[0]) == 0:
                print(f"Warning: No foreground pixels in {img_file}, skipping...")
                continue
            if len(bg_coords[0]) == 0:
                print(f"Warning: No background pixels in {img_file}, skipping...")
                continue
            
            # Sample pixels with stratified sampling
            def sample_pixels(coords, n_samples):
                n_pixels = len(coords[0])
                if n_pixels <= n_samples:
                    indices = range(n_pixels)
                else:
                    indices = random.sample(range(n_pixels), n_samples)
                return [(coords[0][i], coords[1][i]) for i in indices]
            
            fg_samples = sample_pixels(fg_coords, samples_per_class)
            bg_samples = sample_pixels(bg_coords, samples_per_class)
            
            # Process sampled pixels
            for is_fg, samples in enumerate([bg_samples, fg_samples]):
                for x, y in samples:
                    # Validate coordinates (should always be valid due to how we selected them)
                    if 0 <= x < h and 0 <= y < w:
                        features = extract_pixel_features(img, x, y, mask, superpixels)
                        
                        if features is not None:
                            # Create row with metadata
                            row = {
                                'image_name': img_file,
                                'pixel_x': int(x),
                                'pixel_y': int(y),
                                'target': is_fg  # 0 for background, 1 for foreground
                            }
                            
                            # Add features
                            row.update(features)
                            
                            # Add placeholder columns for lower resolution predictions
                            for placeholder in placeholders:
                                row[placeholder] = None
                                
                            dataset_rows.append(row)
        
        # Write batch to CSV
        if dataset_rows:
            df = pd.DataFrame(dataset_rows)
            
            # Ensure consistent column ordering
            cols = ['image_name', 'pixel_x', 'pixel_y', 'target'] + \
                   [col for col in sorted(df.columns) if col not in ['image_name', 'pixel_x', 'pixel_y', 'target']]
            df = df[cols]
            
            # Write to file (create or append)
            if not os.path.exists(output_file):
                df.to_csv(output_file, index=False)
            else:
                df.to_csv(output_file, index=False, mode='a', header=False)
                
            print(f"Saved batch of {len(dataset_rows)} rows to {output_file}")
            dataset_rows = []  # Clear for next batch
    
    print(f"Completed dataset creation for {scale_name} scale.")

# Create datasets for each scale
scales = ['twenty_five', 'fifty', 'original']
create_scale_dataset('twenty_five', 200)
create_scale_dataset('fifty', 400)
create_scale_dataset('original', 600)

print("All datasets created successfully!")