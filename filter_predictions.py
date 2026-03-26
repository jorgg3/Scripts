import os
import shutil
import numpy as np

def filter_with_custom_threshold(
    npz_file_path: str, 
    output_directory: str = "strict_filtered_results", 
    threshold: float = 0.85
):
    """
    Filters images by applying a custom probability threshold to the raw model logits.
    This reduces False Positives by requiring higher statistical confidence for the target class.

    Args:
        npz_file_path (str): Path to the .npz file containing 'logits' and 'file_ids'.
        output_directory (str): Root directory for the separated images.
        threshold (float): Minimum probability [0, 1] required to classify as Cattle.
    """
    print(f"Loading logits from: {npz_file_path}")
    print(f"Applying strict decision threshold: P(Y=Cattle) >= {threshold}")
    
    try:
        data = np.load(npz_file_path)
        logits = data['logits']
        file_ids = data['file_ids']
    except Exception as e:
        print(f"Error loading the .npz file: {e}")
        return

    # Map logits to probabilities
    # Handling both binary (1D) and multiclass (2D) logit shapes gracefully
    if len(logits.shape) == 2 and logits.shape[1] > 1:
        # Numerically stable Softmax for (N, 2) shape
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Assuming class 1 is Cattle
        cattle_probs = probabilities[:, 1] 
    else:
        # Sigmoid for (N, 1) or (N,) shape
        cattle_probs = 1 / (1 + np.exp(-logits.squeeze()))

    cattle_dir = os.path.join(output_directory, "cattle_detected")
    wildlife_dir = os.path.join(output_directory, "native_wildlife")
    
    os.makedirs(cattle_dir, exist_ok=True)
    os.makedirs(wildlife_dir, exist_ok=True)
    
    cattle_count = 0
    wildlife_count = 0

    for prob, filepath in zip(cattle_probs, file_ids):
        filepath = str(filepath)
        filename = os.path.basename(filepath)
        
        # Apply the custom threshold logic
        if prob >= threshold: 
            shutil.copy(filepath, os.path.join(cattle_dir, filename))
            cattle_count += 1
        else:
            shutil.copy(filepath, os.path.join(wildlife_dir, filename))
            wildlife_count += 1
            
    print("-" * 50)
    print("Threshold filtering completed successfully.")
    print(f"Cattle images (High Confidence >= {threshold}): {cattle_count}")
    print(f"Native wildlife images retained:           {wildlife_count}")
    print(f"Total images processed:                    {cattle_count + wildlife_count}")
    print("-" * 50)

if __name__ == "__main__":
    PREDICTION_NPZ_PATH = "weights/Results/Plain/Prueba_Augmentations_ResNet50-0-epoch=10-valid_mac_acc=93.37_predict.npz"
    
    # You can tune this parameter. 0.50 is the default. 
    # 0.85 to 0.95 is recomended to minimize false positives (e.g., misclassified pumas).
    DECISION_THRESHOLD = 0.85
    
    if os.path.exists(PREDICTION_NPZ_PATH):
        filter_with_custom_threshold(
            npz_file_path=PREDICTION_NPZ_PATH, 
            threshold=DECISION_THRESHOLD
        )
    else:
        print(f"Error: The specified file was not found at {PREDICTION_NPZ_PATH}")