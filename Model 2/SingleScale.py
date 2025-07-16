import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('control_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_direct_model():
    """Train a Random Forest directly on original resolution without hierarchical features"""
    
    logger.info("Starting direct model training (no hierarchical features)")
    
    # Check if dataset exists before attempting to load
    original_path = './csv_datasets/original_features.csv'
    if not os.path.exists(original_path):
        logger.error(f"Dataset file not found: {original_path}")
        raise FileNotFoundError(f"Could not find dataset at {original_path}")
    
    # Load the original scale dataset
    logger.info(f"Loading dataset from {original_path}")
    data_original = pd.read_csv(original_path)
    
    # Separate features and target
    meta_columns = ['image_name', 'pixel_x', 'pixel_y', 'target', 'fifty_pred', 'twenty_five_pred']
    X_original = data_original.drop(columns=meta_columns)
    y_original = data_original['target']
    
    logger.info(f"Dataset loaded: {len(data_original)} samples, {X_original.shape[1]} features")
    logger.info(f"Class distribution: {y_original.value_counts().to_dict()}")
    
    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y_original, test_size=0.2, random_state=42, stratify=y_original)
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Train the model (same parameters as hierarchical for fair comparison)
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    logger.info("Evaluating model performance...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='binary')
    auc = roc_auc_score(y_test, probabilities)
    
    logger.info(f"Model performance metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-score: {f1:.4f}")
    logger.info(f"  AUC-ROC: {auc:.4f}")
    
    # Save the model
    model_path = './models/direct_model.pkl'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc,
        "model_path": model_path
    }

if __name__ == "__main__":
    try:
        results = train_direct_model()
        print("Training completed successfully!")
        print(f"Model saved to {results['model_path']}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        print(f"Error: {str(e)}")