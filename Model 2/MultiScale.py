import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import KDTree
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from scipy.ndimage import distance_transform_edt
import time
import logging
from tqdm import tqdm
import multiprocessing
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

# Create necessary directories
os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# Feature extraction function (the real implementation)
def extract_pixel_features(img, x, y, superpixels=None):
    """Extract comprehensive features for a pixel at position (x, y)"""
    h, w = img.shape[:2]
    features = {}
    
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

def process_batch(batch_data, data_lower, model_lower, kdtree_cache, scale_factor, column_name):
    """Process a batch of data to add predictions from lower scale model"""
    results = batch_data.copy()
    
    # Create mappings to lower scale
    mappings = []
    for _, row in batch_data.iterrows():
        img_name = row['image_name']
        x_lower = row['pixel_x'] // scale_factor
        y_lower = row['pixel_y'] // scale_factor
        mappings.append((img_name, x_lower, y_lower))
    
    # Process each mapping
    predictions = []
    for mapping in tqdm(mappings):
        img_name, x_lower, y_lower = mapping
        
        # Try to find exact match first
        query = data_lower[
            (data_lower['image_name'] == img_name) & 
            (data_lower['pixel_x'] == x_lower) & 
            (data_lower['pixel_y'] == y_lower)
        ]
        
        if len(query) > 0:
            # Use exact match
            features = query.drop(columns=['image_name', 'pixel_x', 'pixel_y', 'target']).iloc[0]
            pred_prob = model_lower.predict_proba(features.values.reshape(1, -1))[0, 1]
        else:
            # No exact match, use nearest neighbor
            img_pixels = data_lower[data_lower['image_name'] == img_name]
            
            if len(img_pixels) > 0:
                # Create or retrieve KDTree for this image
                if img_name not in kdtree_cache:
                    pixel_coords = img_pixels[['pixel_x', 'pixel_y']].values
                    kdtree_cache[img_name] = KDTree(pixel_coords)
                
                # Find nearest neighbor
                distances, indices = kdtree_cache[img_name].query([[x_lower, y_lower]], k=1)
                
                # Get closest pixel and its prediction
                closest_idx = indices[0][0]
                closest_pixel = img_pixels.iloc[closest_idx]
                features = closest_pixel.drop(['image_name', 'pixel_x', 'pixel_y', 'target'])
                pred_prob = model_lower.predict_proba(features.values.reshape(1, -1))[0, 1]
            else:
                # If no pixels from this image in dataset, use overall class distribution
                pred_prob = data_lower['target'].mean()
        
        predictions.append(pred_prob)
    
    # Add predictions to results
    results[column_name] = predictions
    return results

# Parallel feature extraction for image
def extract_features_parallel(args):
    """Extract features for a subset of pixels"""
    img, coords, superpixels = args
    features_list = []
    
    for x, y in coords:
        features = extract_pixel_features(img, x, y, superpixels)
        features_list.append(features)
    
    return features_list

def segment_image(image_path, output_path=None, visualization_path=None):
    """
    Apply the hierarchical segmentation model to an image
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save segmentation mask
        visualization_path: Optional path to save visualization
    
    Returns:
        Segmentation mask (numpy array)
    """
    logger.info(f"Starting segmentation of {image_path}")
    start_time = time.time()
    
    # Load models
    logger.info("Loading models...")
    model_25 = joblib.load('./models/twenty_five_model.pkl')
    model_50 = joblib.load('./models/fifty_model.pkl')
    model_original = joblib.load('./models/original_model.pkl')
    
    # Get feature names (needed for prediction)
    data_25 = pd.read_csv('./csv_datasets/twenty_five_features.csv', nrows=1)
    feature_cols_25 = [c for c in data_25.columns if c not in ['image_name', 'pixel_x', 'pixel_y', 'target']]
    
    data_50 = pd.read_csv('./csv_datasets/fifty_features_augmented.csv', nrows=1)
    feature_cols_50 = [c for c in data_50.columns if c not in ['image_name', 'pixel_x', 'pixel_y', 'target']]
    
    data_original = pd.read_csv('./csv_datasets/original_features_augmented.csv', nrows=1)
    feature_cols_original = [c for c in data_original.columns if c not in ['image_name', 'pixel_x', 'pixel_y', 'target']]
    
    # Load image
    logger.info("Loading and preprocessing image...")
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img.shape[:2]
    
    # Create downscaled versions
    fifty_img = cv2.resize(original_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    twenty_five_img = cv2.resize(original_img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    
    logger.info("Computing superpixels...")
    # Compute superpixels for each scale
    superpixels_25 = slic(twenty_five_img, n_segments=50, compactness=10, sigma=1, start_label=1)
    superpixels_50 = slic(fifty_img, n_segments=100, compactness=10, sigma=1, start_label=1)
    superpixels_original = slic(original_img, n_segments=200, compactness=10, sigma=1, start_label=1)
    
    # Step 1: Process twenty_five scale
    logger.info("Processing twenty_five scale image...")
    h_25, w_25 = twenty_five_img.shape[:2]
    
    # Create mesh grid for twenty_five scale (process every pixel)
    y_25, x_25 = np.meshgrid(np.arange(w_25), np.arange(h_25))
    coords_25 = list(zip(x_25.flatten(), y_25.flatten()))
    
    # Split coordinates for parallel processing
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(coords_25) // num_cores
    chunks_25 = [coords_25[i:i + chunk_size] for i in range(0, len(coords_25), chunk_size)]
    
    # Prepare arguments for parallel processing
    args_25 = [(twenty_five_img, chunk, superpixels_25) for chunk in chunks_25]
    
    # Extract features in parallel
    logger.info(f"Extracting features using {num_cores} cores...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_25 = pool.map(extract_features_parallel, args_25)
    
    # Flatten results
    features_list_25 = [item for sublist in results_25 for item in sublist]
    
    # Create DataFrame
    logger.info("Creating twenty_five scale predictions...")
    df_25 = pd.DataFrame(features_list_25)
    
    # Make predictions
    X_25 = df_25[feature_cols_25].fillna(0)  # Handle any missing features
    probs_25 = model_25.predict_proba(X_25)[:, 1]
    
    # Create prediction grid
    pred_grid_25 = np.zeros((h_25, w_25))
    for (x, y), prob in zip(coords_25, probs_25):
        pred_grid_25[x, y] = prob
    
    # Step 2: Process fifty scale using twenty_five predictions
    logger.info("Processing fifty scale image...")
    h_50, w_50 = fifty_img.shape[:2]
    
    # Create mesh grid for fifty scale
    stride_50 = 2  # Process every 2nd pixel for speed (adjust based on your needs)
    y_50, x_50 = np.meshgrid(
        np.arange(0, w_50, stride_50),
        np.arange(0, h_50, stride_50)
    )
    coords_50 = list(zip(x_50.flatten(), y_50.flatten()))
    
    # Split coordinates for parallel processing
    chunks_50 = [coords_50[i:i + chunk_size] for i in range(0, len(coords_50), chunk_size)]
    args_50 = [(fifty_img, chunk, superpixels_50) for chunk in chunks_50]
    
    # Extract features in parallel
    logger.info("Extracting fifty scale features...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_50 = pool.map(extract_features_parallel, args_50)
    
    # Flatten results
    features_list_50 = [item for sublist in results_50 for item in sublist]
    
    # Create DataFrame
    df_50 = pd.DataFrame(features_list_50)
    
    # Add twenty_five scale predictions
    for i, (x, y) in enumerate(coords_50):
        # Map to twenty_five scale
        x_25 = x // 2
        y_25 = y // 2
        
        # Get prediction from twenty_five scale
        if 0 <= x_25 < h_25 and 0 <= y_25 < w_25:
            df_50.at[i, 'twenty_five_pred'] = pred_grid_25[x_25, y_25]
        else:
            df_50.at[i, 'twenty_five_pred'] = 0.5
    
    # Make predictions
    X_50 = df_50[feature_cols_50].fillna(0)
    probs_50 = model_50.predict_proba(X_50)[:, 1]
    
    # Create prediction grid (initialize with 0.5 and fill in predictions)
    pred_grid_50 = np.ones((h_50, w_50)) * 0.5
    for (x, y), prob in zip(coords_50, probs_50):
        pred_grid_50[x, y] = prob
    
    # Interpolate to fill in missing values
    x_coords, y_coords = np.mgrid[0:h_50, 0:w_50]
    points = np.array(coords_50)
    values = np.array(probs_50)
    from scipy.interpolate import griddata
    pred_grid_50 = griddata(points, values, (x_coords, y_coords), method='nearest')
    
    # Step 3: Process original scale using both lower scale predictions
    logger.info("Processing original scale image...")
    
    # Use a larger stride for original image to speed up processing
    stride_orig = 4
    RandomForest()
    y_orig, x_orig = np.meshgrid(
        np.arange(0, w, stride_orig),
        np.arange(0, h, stride_orig)
    )
    coords_orig = list(zip(x_orig.flatten(), y_orig.flatten()))
    
    # Split coordinates for parallel processing
    chunk_size_orig = max(1, len(coords_orig) // num_cores)
    chunks_orig = [coords_orig[i:i + chunk_size_orig] for i in range(0, len(coords_orig), chunk_size_orig)]
    args_orig = [(original_img, chunk, superpixels_original) for chunk in chunks_orig]
    
    # Extract features in parallel
    logger.info("Extracting original scale features...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_orig = pool.map(extract_features_parallel, args_orig)
    
    # Flatten results
    features_list_orig = [item for sublist in results_orig for item in sublist]
    
    # Create DataFrame
    df_orig = pd.DataFrame(features_list_orig)
    
    # Add lower scale predictions
    for i, (x, y) in enumerate(coords_orig):
        # Map to fifty scale
        x_50 = x // 2
        y_50 = y // 2
        
        # Map to twenty_five scale
        x_25 = x // 4
        y_25 = y // 4
        
        # Get prediction from fifty scale
        if 0 <= x_50 < h_50 and 0 <= y_50 < w_50:
            df_orig.at[i, 'fifty_pred'] = pred_grid_50[x_50, y_50]
        else:
            df_orig.at[i, 'fifty_pred'] = 0.5
        
        # Get prediction from twenty_five scale
        if 0 <= x_25 < h_25 and 0 <= y_25 < w_25:
            df_orig.at[i, 'twenty_five_pred'] = pred_grid_25[x_25, y_25]
        else:
            df_orig.at[i, 'twenty_five_pred'] = 0.5
    
    # Make predictions
    X_orig = df_orig[feature_cols_original].fillna(0)
    probs_orig = model_original.predict_proba(X_orig)[:, 1]
    
    # Create final segmentation mask
    segmentation_probs = np.ones((h, w)) * 0.5  # Initialize with uncertainty
    for (x, y), prob in zip(coords_orig, probs_orig):
        segmentation_probs[x, y] = prob
    
    # Interpolate to fill gaps
    x_coords, y_coords = np.mgrid[0:h, 0:w]
    points = np.array(coords_orig)
    values = np.array(probs_orig)
    segmentation_probs = griddata(points, values, (x_coords, y_coords), method='nearest')
    
    # Apply threshold to get binary mask
    segmentation_mask = (segmentation_probs > 0.5).astype(np.uint8)
    
    # Optional: Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel)
    
    # Save output if requested
    if output_path:
        cv2.imwrite(output_path, segmentation_mask * 255)
        logger.info(f"Saved segmentation mask to {output_path}")
    
    # Create visualization if requested
    if visualization_path:
        # Create a colored overlay
        vis_img = original_img.copy()
        mask_rgb = np.zeros_like(original_img)
        mask_rgb[segmentation_mask == 1] = [0, 255, 0]  # Green for foreground
        alpha = 0.5
        vis_img = cv2.addWeighted(mask_rgb, alpha, vis_img, 1 - alpha, 0)
        
        # Draw contours around the segmentation for better visibility
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)
        
        # Convert back to BGR for saving
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(visualization_path, vis_img)
        logger.info(f"Saved visualization to {visualization_path}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Segmentation completed in {elapsed_time:.2f} seconds")
    
    return segmentation_mask

def main():
    """Main pipeline execution"""
    start_time = time.time()
    logger.info("Starting hierarchical ML segmentation pipeline")
    
    #=====================================================================
    # PART 1: DATA PREPARATION
    #=====================================================================
    
    logger.info("=" * 80)
    logger.info("HIERARCHICAL SEGMENTATION PIPELINE")
    logger.info("=" * 80)
    
    # Check dataset files
    csv_datasets_path = './csv_datasets/'
    data_paths = {
        'twenty_five': os.path.join(csv_datasets_path, 'twenty_five_features.csv'),
        'fifty': os.path.join(csv_datasets_path, 'fifty_features.csv'),
        'original': os.path.join(csv_datasets_path, 'original_features.csv')
    }
    
    for scale, path in data_paths.items():
        if not os.path.exists(path):
            logger.error(f"Dataset for {scale} scale not found at {path}")
            return
        else:
            file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
            logger.info(f"Found {scale} dataset: {path} ({file_size:.2f} MB)")
    
    #=====================================================================
    # PART 2: TRAINING TWENTY_FIVE SCALE MODEL
    #=====================================================================
    
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: TRAINING TWENTY_FIVE SCALE MODEL")
    logger.info("=" * 50)
    
    # Load twenty_five dataset
    logger.info("Loading twenty_five scale dataset...")
    data_25 = pd.read_csv(data_paths['twenty_five'])
    from sklearn.ensemble import RandomForestClassifier
    logger.info(f"Loaded {len(data_25)} samples from twenty_five scale dataset")
    
    # Separate features and target
    logger.info("Preparing features and target...")
    meta_columns = ['image_name', 'pixel_x', 'pixel_y', 'target']
    X_25 = data_25.drop(columns=meta_columns)
    y_25 = data_25['target']
    
    # Log feature information
    logger.info(f"Number of features: {X_25.shape[1]}")
    logger.info(f"Class distribution: {y_25.value_counts().to_dict()}")
    
    # Split data for training and evaluation
    X_train_25, X_test_25, y_train_25, y_test_25 = train_test_split(
        X_25, y_25, test_size=0.2, random_state=42, stratify=y_25)
    
    logger.info(f"Training set: {X_train_25.shape[0]} samples")
    logger.info(f"Testing set: {X_test_25.shape[0]} samples")
    
    # Train the model
    logger.info("Training Random Forest model on twenty_five scale data...")
    model_25 = RandomForestClassifier(
        n_estimators=200, 
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model_25.fit(X_train_25, y_train_25)
    
    # Evaluate the model
    logger.info("Evaluating twenty_five scale model...")
    predictions_25 = model_25.predict(X_test_25)
    probabilities_25 = model_25.predict_proba(X_test_25)[:, 1]
    
    accuracy_25 = accuracy_score(y_test_25, predictions_25)
    precision_25, recall_25, f1_25, _ = precision_recall_fscore_support(
        y_test_25, predictions_25, average='binary')
    auc_25 = roc_auc_score(y_test_25, probabilities_25)
    
    logger.info(f"Twenty-five scale model performance:")
    logger.info(f"  Accuracy: {accuracy_25:.4f}")
    logger.info(f"  Precision: {precision_25:.4f}")
    logger.info(f"  Recall: {recall_25:.4f}")
    logger.info(f"  F1-score: {f1_25:.4f}")
    logger.info(f"  AUC-ROC: {auc_25:.4f}")
    
    # Save model
    model_path_25 = './models/twenty_five_model.pkl'
    joblib.dump(model_25, model_path_25)
    logger.info(f"Model saved to {model_path_25}")
    
    #=====================================================================
    # PART 3: AUGMENTING FIFTY SCALE DATASET WITH TWENTY_FIVE PREDICTIONS
    #=====================================================================
    
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: AUGMENTING FIFTY SCALE DATASET")
    logger.info("=" * 50)
    
    # Load fifty scale dataset
    logger.info("Loading fifty scale dataset...")
    data_50 = pd.read_csv(data_paths['fifty'])
    logger.info(f"Loaded {len(data_50)} samples from fifty scale dataset")
    
    # Process fifty scale dataset in batches
    logger.info("Augmenting fifty scale dataset with twenty_five predictions...")
    batch_size = 10000
    output_path_50 = './csv_datasets/fifty_features_augmented.csv'
    
    # Remove output file if it exists
    if os.path.exists(output_path_50):
        os.remove(output_path_50)
    
    # Initialize KDTree cache
    kdtree_cache_25 = {}
    
    # Split data for parallel processing
    num_batches = (len(data_50) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data_50))
        logger.info(f"Processing batch {i+1}/{num_batches} ({start_idx}-{end_idx})")
        
        batch = data_50.iloc[start_idx:end_idx].copy()
        
        # Process in parallel if large enough
        if len(batch) > 1000:
            # Split batch into chunks for parallel processing
            num_chunks = min(multiprocessing.cpu_count(), (len(batch) + 999) // 1000)
            chunk_size = (len(batch) + num_chunks - 1) // num_chunks
            
            batch_chunks = [batch.iloc[j:j+chunk_size] for j in range(0, len(batch), chunk_size)]
            
            # Process each chunk in parallel
            with multiprocessing.Pool(processes=num_chunks) as pool:
                process_func = partial(
                    process_batch, 
                    data_lower=data_25,
                    model_lower=model_25,
                    kdtree_cache=kdtree_cache_25,
                    scale_factor=2,
                    column_name='twenty_five_pred'
                )
                results = pool.map(process_func, batch_chunks)
            
            # Combine results
            batch = pd.concat(results, ignore_index=True)
        else:
            # Process small batch directly
            batch = process_batch(
                batch, data_25, model_25, kdtree_cache_25, 2, 'twenty_five_pred')
        
        # Save batch
        if i == 0:
            batch.to_csv(output_path_50, index=False)
        else:
            batch.to_csv(output_path_50, mode='a', header=False, index=False)
    
    logger.info(f"Fifty scale dataset augmented and saved to {output_path_50}")
    
    #=====================================================================
    # PART 4: TRAINING FIFTY SCALE MODEL
    #=====================================================================
    
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: TRAINING FIFTY SCALE MODEL")
    logger.info("=" * 50)
    
    # Load augmented fifty scale dataset
    logger.info("Loading augmented fifty scale dataset...")
    data_50_augmented = pd.read_csv(output_path_50)
    logger.info(f"Loaded {len(data_50_augmented)} samples from augmented fifty scale dataset")
    
    # Separate features and target
    logger.info("Preparing features and target...")
    X_50 = data_50_augmented.drop(columns=meta_columns)
    y_50 = data_50_augmented['target']
    
    # Log feature information
    logger.info(f"Number of features: {X_50.shape[1]}")
    logger.info(f"Class distribution: {y_50.value_counts().to_dict()}")
    
    # Split data for training and evaluation
    X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(
        X_50, y_50, test_size=0.2, random_state=42, stratify=y_50)
    
    logger.info(f"Training set: {X_train_50.shape[0]} samples")
    logger.info(f"Testing set: {X_test_50.shape[0]} samples")
    
    # Train the model
    logger.info("Training Random Forest model on fifty scale data...")
    model_50 = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model_50.fit(X_train_50, y_train_50)
    
    # Evaluate the model
    logger.info("Evaluating fifty scale model...")
    predictions_50 = model_50.predict(X_test_50)
    probabilities_50 = model_50.predict_proba(X_test_50)[:, 1]
    
    accuracy_50 = accuracy_score(y_test_50, predictions_50)
    precision_50, recall_50, f1_50, _ = precision_recall_fscore_support(
        y_test_50, predictions_50, average='binary')
    auc_50 = roc_auc_score(y_test_50, probabilities_50)
    
    logger.info(f"Fifty scale model performance:")
    logger.info(f"  Accuracy: {accuracy_50:.4f}")
    logger.info(f"  Precision: {precision_50:.4f}")
    logger.info(f"  Recall: {recall_50:.4f}")
    logger.info(f"  F1-score: {f1_50:.4f}")
    logger.info(f"  AUC-ROC: {auc_50:.4f}")
    
    # Save model
    model_path_50 = './models/fifty_model.pkl'
    joblib.dump(model_50, model_path_50)
    logger.info(f"Model saved to {model_path_50}")
    
    #=====================================================================
    # PART 5: AUGMENTING ORIGINAL SCALE DATASET WITH HIERARCHICAL PREDICTIONS
    #=====================================================================
    
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: AUGMENTING ORIGINAL SCALE DATASET")
    logger.info("=" * 50)
    
    # Load original scale dataset
    logger.info("Loading original scale dataset...")
    data_original = pd.read_csv(data_paths['original'])
    logger.info(f"Loaded {len(data_original)} samples from original scale dataset")
    
    # Initialize KDTree cache for fifty scale
    kdtree_cache_50 = {}
    
    # Process original scale dataset in batches
    logger.info("Augmenting original scale dataset with hierarchical predictions...")
    output_path_original = './csv_datasets/original_features_augmented.csv'
    
    # Remove output file if it exists
    if os.path.exists(output_path_original):
        os.remove(output_path_original)
    
    num_batches = (len(data_original) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data_original))
        logger.info(f"Processing batch {i+1}/{num_batches} ({start_idx}-{end_idx})")
        
        batch = data_original.iloc[start_idx:end_idx].copy()
        
        # First add twenty_five predictions
        batch = process_batch(
            batch, data_25, model_25, kdtree_cache_25, 4, 'twenty_five_pred')
        
        # Then add fifty predictions
        batch = process_batch(
            batch, data_50_augmented, model_50, kdtree_cache_50, 2, 'fifty_pred')
        
        # Save batch
        if i == 0:
            batch.to_csv(output_path_original, index=False)
        else:
            batch.to_csv(output_path_original, mode='a', header=False, index=False)
    
    logger.info(f"Original scale dataset augmented and saved to {output_path_original}")
    
    #=====================================================================
    # PART 6: TRAINING FINAL ORIGINAL SCALE MODEL
    #=====================================================================
    
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 5: TRAINING FINAL MODEL")
    logger.info("=" * 50)
    
    # Load augmented original scale dataset
    logger.info("Loading augmented original scale dataset...")
    data_original_augmented = pd.read_csv(output_path_original)
    logger.info(f"Loaded {len(data_original_augmented)} samples from augmented original scale dataset")
    
    # Separate features and target
    logger.info("Preparing features and target...")
    X_original = data_original_augmented.drop(columns=meta_columns)
    y_original = data_original_augmented['target']
    
    # Log feature information
    logger.info(f"Number of features: {X_original.shape[1]}")
    logger.info(f"Class distribution: {y_original.value_counts().to_dict()}")
    
    # Split data for training and evaluation
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
        X_original, y_original, test_size=0.2, random_state=42, stratify=y_original)
    
    logger.info(f"Training set: {X_train_original.shape[0]} samples")
    logger.info(f"Testing set: {X_test_original.shape[0]} samples")
    
    # Train the model
    logger.info("Training Random Forest model on original scale data...")
    model_original = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model_original.fit(X_train_original, y_train_original)
    
    # Evaluate the model
    logger.info("Evaluating original scale model...")
    predictions_original = model_original.predict(X_test_original)
    probabilities_original = model_original.predict_proba(X_test_original)[:, 1]
    
    accuracy_original = accuracy_score(y_test_original, predictions_original)
    precision_original, recall_original, f1_original, _ = precision_recall_fscore_support(
        y_test_original, predictions_original, average='binary')
    auc_original = roc_auc_score(y_test_original, probabilities_original)
    
    logger.info(f"Original scale model performance:")
    logger.info(f"  Accuracy: {accuracy_original:.4f}")
    logger.info(f"  Precision: {precision_original:.4f}")
    logger.info(f"  Recall: {recall_original:.4f}")
    logger.info(f"  F1-score: {f1_original:.4f}")
    logger.info(f"  AUC-ROC: {auc_original:.4f}")
    
    # Save model
    model_path_original = './models/original_model.pkl'
    joblib.dump(model_original, model_path_original)
    logger.info(f"Model saved to {model_path_original}")
    
    #=====================================================================
    # PART 7: FEATURE IMPORTANCE ANALYSIS
    #=====================================================================
    
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 6: FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 50)
    
    # Get feature importances from the final model
    feature_importances = model_original.feature_importances_
    feature_names = X_original.columns
    
    # Create DataFrame for feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Print top 20 features
    logger.info("Top 20 most important features:")
    logger.info(importance_df.head(20).to_string(index=False))
    
    # Highlight importance of hierarchical features
    hierarchical_features = importance_df[
        importance_df['Feature'].isin(['twenty_five_pred', 'fifty_pred'])
    ]
    logger.info("\nImportance of hierarchical features:")
    logger.info(hierarchical_features.to_string(index=False))
    
    # Save feature importance
    importance_df.to_csv('./results/feature_importance.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'].head(20)[::-1], importance_df['Importance'].head(20)[::-1])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('./results/feature_importance.png')
    
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 50)
    logger.info(f"PIPELINE COMPLETED IN {total_time:.2f} SECONDS")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()