import numpy as np
from collections import Counter
import joblib

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        # Convert pandas objects to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        self.tree = self._grow_tree(X, y)
        return self

    def predict(self, X):
        # Convert pandas objects to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or 
            num_labels == 1 or 
            num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return leaf_value

        # Find best split
        feat_idxs = np.random.choice(num_features, num_features, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # Split data
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        # Handle edge case: if split didn't work
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return leaf_value

        # Recursively grow left and right children
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return (best_feat, best_thresh, left, right)

    def _traverse_tree(self, x, node):
        # Leaf node (contains the class prediction)
        if not isinstance(node, tuple):
            return node[0][0]
        
        # Decision node
        feature_idx, threshold, left, right = node
        
        if x[feature_idx] <= threshold:
            return self._traverse_tree(x, left)
        return self._traverse_tree(x, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
                    
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        # Check if split is valid
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Information gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        # Convert to integers for bincount
        y_int = y.astype(int)
        hist = np.bincount(y_int)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)

class RandomForest:
    def __init__(self, n_estimators=200, max_depth=20, min_samples_split=5, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X, y):
        # Convert pandas to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        self.trees = []
        
        # Use parallel processing if n_jobs != 1
        if self.n_jobs != 1:
            self.trees = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._fit_tree)(X, y) 
                for _ in range(self.n_estimators)
            )
        else:
            # Sequential processing
            for _ in range(self.n_estimators):
                self.trees.append(self._fit_tree(X, y))
                
        return self

    def _fit_tree(self, X, y):
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        # Bootstrap sampling
        X_sample, y_sample = self._bootstrap_sample(X, y)
        # Fit the tree
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        # Convert pandas to numpy
        if hasattr(X, 'values'):
            X = X.values
            
        # Get predictions from all trees
        if self.n_jobs != 1:
            tree_preds = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(tree.predict)(X) for tree in self.trees
            )
            tree_preds = np.array(tree_preds)
        else:
            tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to have predictions per sample
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Majority voting
        y_pred = []
        for predictions in tree_preds:
            counter = Counter(predictions)
            y_pred.append(counter.most_common(1)[0][0])
            
        return np.array(y_pred)

    def _bootstrap_sample(self, X, y):
        # Get number of samples
        n_samples = X.shape[0]
        # Sample with replacement
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]