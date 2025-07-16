import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, hog

def get_lbp_feature(img):
    param = [[8,1], [24,3], [72, 9]]
    lbp_features = []
    for p, r in param:
        lbp = local_binary_pattern(img, p, r)
        lbp_features.append(lbp)
    return lbp_features
def get_quest_feature(image, window_size=5):
    H, W = image.shape
    
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    Ixx = gx * gx
    Iyy = gy * gy
    Ixy = gx * gy
    
    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)   
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    
    lamda1 = 0.5 * (Sxx + Syy + np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2))
    lamda2 = 0.5 * (Sxx + Syy - np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2))
    
    quest_features = np.concatenate([lamda1, lamda2]).reshape(H*W, 2)
    return quest_features

def get_hog_feature(img):
    H, W = img.shape
    pixels_per_cell = [(8,8), (16,16)]
    cells_per_block = [(2,2), (4,4)]
    expected_size = H * W
    hog_features = []
    for ppc, cpb in zip(pixels_per_cell, cells_per_block):
        hog_f = hog(
            img, 
            pixels_per_cell=ppc, 
            cells_per_block=cpb, 
            block_norm='L2-Hys',
            feature_vector=True
        )
        if hog_f.size < expected_size:
            hog_f = np.tile(hog_f, (expected_size // hog_f.size + 1))[:expected_size]
        elif hog_f.size > expected_size:
            hog_f = hog_f[:expected_size]
        
        hog_f = hog_f.reshape(-1, 1)
        hog_features.append(hog_f)
    return hog_features
