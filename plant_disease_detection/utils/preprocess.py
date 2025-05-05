import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess image for model prediction
    Args:
        image_path: Path to the image file
        target_size: Tuple of (height, width) for resizing
    Returns:
        Preprocessed image array
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img 