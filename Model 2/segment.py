import os
import numpy as np
import pandas as pd
import joblib
import cv2
import multiprocessing
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Parallel feature extraction for image
def extract_features_parallel(args):
    """Extract features for a subset of pixels"""
    img, coords, superpixels = args
    features_list = []
    
    for x, y in coords:
        features = extract_pixel_features(img, x, y, superpixels)
        features_list.append(features)
    
    return features_list

def segment_image_multiscale(image_path, output_path=None, visualization_path=None):
    """
    Apply hierarchical multi-scale ML segmentation to an image
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save segmentation mask
        visualization_path: Optional path to save visualization
    
    Returns:
        Segmentation mask (numpy array)
    """
    logger.info(f"Starting MultiScale segmentation of {image_path}")
    start_time = time.time()
    
    # Load models
    logger.info("Loading multi-scale models...")
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
    stride_50 = 2  # Process every 2nd pixel for speed
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
    
    # Create prediction grid and interpolate
    pred_grid_50 = np.ones((h_50, w_50)) * 0.5
    for (x, y), prob in zip(coords_50, probs_50):
        pred_grid_50[x, y] = prob
    
    # Interpolate to fill in missing values
    x_coords, y_coords = np.mgrid[0:h_50, 0:w_50]
    points = np.array(coords_50)
    values = np.array(probs_50)
    pred_grid_50 = griddata(points, values, (x_coords, y_coords), method='nearest')
    
    # Step 3: Process original scale using both lower scale predictions
    logger.info("Processing original scale image...")
    
    # Use a larger stride for original image to speed up processing
    stride_orig = 4
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
        
    # Create and save outputs
    if output_path:
        cv2.imwrite(output_path, segmentation_mask * 255)
        logger.info(f"Saved segmentation mask to {output_path}")
    
    if visualization_path:
        # Create visualization
        vis_img = original_img.copy()
        mask_rgb = np.zeros_like(original_img)
        mask_rgb[segmentation_mask == 1] = [0, 255, 0]  # Green for foreground
        alpha = 0.5
        vis_img = cv2.addWeighted(mask_rgb, alpha, vis_img, 1 - alpha, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)
        
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(visualization_path, vis_img)
        logger.info(f"Saved visualization to {visualization_path}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"MultiScale segmentation completed in {elapsed_time:.2f} seconds")
    
    return segmentation_mask

def segment_image_singlescale(image_path, output_path=None, visualization_path=None):
    """
    Apply direct single-scale ML segmentation to an image
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save segmentation mask
        visualization_path: Optional path to save visualization
    
    Returns:
        Segmentation mask (numpy array)
    """
    logger.info(f"Starting SingleScale segmentation of {image_path}")
    start_time = time.time()
    
    # Load direct model
    logger.info("Loading SingleScale model...")
    model_direct = joblib.load('./models/direct_model.pkl')
    
    # Load feature columns for the direct model
    data_original = pd.read_csv('./csv_datasets/original_features.csv', nrows=1)
    feature_cols = [c for c in data_original.columns if c not in ['image_name', 'pixel_x', 'pixel_y', 'target', 'fifty_pred', 'twenty_five_pred']]
    
    # Load image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img.shape[:2]
    
    # Compute superpixels
    logger.info("Computing superpixels...")
    superpixels = slic(original_img, n_segments=200, compactness=10, sigma=1, start_label=1)
    
    # Sample pixels with stride for efficiency
    stride = 4
    y_coords, x_coords = np.meshgrid(
        np.arange(0, w, stride),
        np.arange(0, h, stride)
    )
    coords = list(zip(x_coords.flatten(), y_coords.flatten()))
    
    # Extract features in parallel
    logger.info("Extracting features...")
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, len(coords) // num_cores)
    chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
    args = [(original_img, chunk, superpixels) for chunk in chunks]
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(extract_features_parallel, args)
    
    # Flatten results
    features_list = [item for sublist in results for item in sublist]
    
    # Create DataFrame and predict
    logger.info("Making predictions...")
    df = pd.DataFrame(features_list)
    X = df[feature_cols].fillna(0)
    probs = model_direct.predict_proba(X)[:, 1]
    
    # Create prediction grid and interpolate
    segmentation_probs = np.ones((h, w)) * 0.5
    for (x, y), prob in zip(coords, probs):
        segmentation_probs[x, y] = prob
    
    # Fill in gaps
    logger.info("Interpolating results...")
    x_grid, y_grid = np.mgrid[0:h, 0:w]
    points = np.array(coords)
    values = np.array(probs)
    segmentation_probs = griddata(points, values, (x_grid, y_grid), method='nearest')

    # Apply threshold to get binary mask
    segmentation_mask = (segmentation_probs > 0.5).astype(np.uint8)
        
    # Save outputs if requested
    if output_path:
        cv2.imwrite(output_path, segmentation_mask * 255)
        logger.info(f"Saved segmentation mask to {output_path}")
    
    if visualization_path:
        # Create visualization
        vis_img = original_img.copy()
        mask_rgb = np.zeros_like(original_img)
        mask_rgb[segmentation_mask == 1] = [0, 255, 0]  # Green for foreground
        alpha = 0.5
        vis_img = cv2.addWeighted(mask_rgb, alpha, vis_img, 1 - alpha, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)
        
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(visualization_path, vis_img)
        logger.info(f"Saved visualization to {visualization_path}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"SingleScale segmentation completed in {elapsed_time:.2f} seconds")
    
    return segmentation_mask

# Wrapper function that calls the appropriate method
def segment_image(image_path, output_path=None, visualization_path=None, method="multiscale"):
    """
    Segment image using the specified method
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save segmentation mask
        visualization_path: Optional path to save visualization
        method: Segmentation method ('multiscale' or 'singlescale')
    
    Returns:
        Segmentation mask (numpy array)
    """
    if method.lower() == "multiscale":
        return segment_image_multiscale(image_path, output_path, visualization_path)
    else:
        return segment_image_singlescale(image_path, output_path, visualization_path)

import sys
from segment import segment_image

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: segment.py input_path output_path [visualization_path] [method]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    visualization_path = sys.argv[3] if len(sys.argv) > 3 else None
    method = sys.argv[4] if len(sys.argv) > 4 else "multiscale"
    
    try:
        segment_image(input_path, output_path, visualization_path, method)
        print("Segmentation completed successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)