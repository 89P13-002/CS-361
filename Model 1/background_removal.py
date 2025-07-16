import cv2
from feature_extraction import get_lbp_feature,get_quest_feature,get_hog_feature
import numpy as np
from sklearn.utils import shuffle
from skimage.color import rgb2gray
from hp_tuning import hyperparameter_tuning
from logistic_regression import LogisticRegression
def remove_background_with_box(img : np.ndarray,bboxes : list,max_samples : int = 5000,morph_kernel_size : int = 5,max_hyp_tuning_iter : int = 50,features : list = [])-> np.ndarray :
    H,W,C = img.shape
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box")
        img_converted = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
        H,W,C = img_converted.shape
        
        # foreground and background from bounding box
        mask = np.zeros((H,W),dtype = np.uint8)
        for x1,y1,x2,y2 in bboxes:
            mask[y1:y2,x1:x2] = 1
        X = img_converted.reshape(-1,C)
        y = mask.reshape(-1)
        
        feature_list = []
        if 'lbp' in features:
            gray_img = rgb2gray(img_converted)
            lbp_feats = get_lbp_feature(gray_img)
            lbp_feats = [feat.reshape(-1, 1) for feat in lbp_feats]
            feature_list.extend(lbp_feats)
        if 'quest' in features:
            gray_img = cv2.cvtColor(img_converted, cv2.COLOR_RGB2GRAY)
            quest_feat = get_quest_feature(gray_img)
            feature_list.append(quest_feat)
        if 'hog' in features:
            gray_img = rgb2gray(img_converted)
            hog_feats = get_hog_feature(gray_img)
            feature_list.extend(hog_feats)
        
        if feature_list:
            additional_features = np.concatenate(feature_list, axis=1)
            X = np.hstack([X, additional_features])
        else:
            additional_features = np.empty((X.shape[0], 0))
        
        # Sample foreground and background pixels
        bg_indices = np.where(y == 0)[0]
        fg_indices = np.where(y == 1)[0]
        
        if (8*max_samples) < len(bg_indices):
            bg_idx = np.random.choice(bg_indices, 8*max_samples, replace=False)
        else:
            bg_idx = bg_indices
            
        if len(fg_indices) > 0:
            rows = fg_indices // W
            cols = fg_indices % W
            fg_weights = np.zeros_like(rows,dtype=np.float32)
            
            
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                in_bbox = (rows >= y1) & (rows < y2) & (cols >= x1) & (cols < x2)
                if np.any(in_bbox):
                    center_row = (y1 + y2) / 2.0
                    center_col = (x1 + x2) / 2.0
                    sigma = min((y2 - y1) / 4.0, (x2 - x1) / 4.0)
                    distances = np.sqrt((rows[in_bbox] - center_row) ** 2 + (cols[in_bbox] - center_col) ** 2)
                    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
                    fg_weights[in_bbox] = weights
                    
                    
            if fg_weights.sum() > 0:
                fg_prob = fg_weights / fg_weights.sum()
            else:
                fg_prob = np.ones_like(fg_weights) / len(fg_weights)
                    
            if max_samples < len(fg_indices):
                fg_idx = np.random.choice(fg_indices, max_samples, replace=False, p=fg_prob)
            else:
                fg_idx = fg_indices
        else:
            fg_idx = np.array([])
        
        idx = np.concatenate([bg_idx, fg_idx])
        X_sample = X[idx]   
        y_sample = y[idx]
        
        X_sample, y_sample = shuffle(X_sample, y_sample,random_state=42)
        
        # Do hyperparameter tuning
        best_params = hyperparameter_tuning(X_sample, y_sample, max_hyp_tuning_iter)
        print(f"Best parameters: {best_params}")
        
        model = LogisticRegression(**best_params, class_weight={0: 3, 1: 1})
        model.fit(X_sample, y_sample)
        
        print(f"Model accuracy on training data: {model.score(X_sample, y_sample)}")
        
        # Predict on the entire image
        X_full = np.hstack([img_converted.reshape(-1, C), additional_features])
        y_pred = model.predict(X_full)
        
        mask = y_pred.reshape(H, W)
        mask = mask.astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
                