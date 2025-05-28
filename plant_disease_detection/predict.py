import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from utils.segmentation import LeafSegmenter

class PlantDiseasePredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
            
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.segmenter = LeafSegmenter()
            self.class_names = ['healthy', 'powdery', 'rust']
            self.img_size = (256, 256)  # Changed to match model's expected input size
            # Threshold for considering an image as containing leaves
            self.leaf_threshold = 0.1  # 10% of image area must be green
            # Confidence thresholds for disease classification
            self.confidence_thresholds = {
                'healthy': 0.65,  # Higher threshold for healthy classification
                'powdery': 0.45,  # Lower threshold for powdery mildew
                'rust': 0.50     # Moderate threshold for rust
            }
        except Exception as e:
            raise RuntimeError(f"Predictor initialization failed: {str(e)}")
    
    def _is_leaf_image(self, image_norm):
        """Check if the image contains significant green areas (likely leaves)"""
        # Convert to HSV
        hsv = cv2.cvtColor((image_norm * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Create green mask
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green area
        green_pixels = np.count_nonzero(green_mask)
        total_pixels = green_mask.size
        green_percentage = green_pixels / total_pixels
        
        return green_percentage > self.leaf_threshold
    
    def _preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            return img / 255.0
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {str(e)}")
    
    def predict(self, image_path, leaf_index=None, explain=False, all_leaves=False):
        """
        Predict plant disease from an image
        Args:
            image_path: Path to the image file
            leaf_index: Optional index of specific leaf to analyze (0-based)
            explain: Whether to show detailed explanations
            all_leaves: Whether to analyze all leaves (default: False)
        """
        try:
            # Load and preprocess image
            image_norm = self._preprocess_image(image_path)
            
            # First check if this looks like a leaf image at all
            if not self._is_leaf_image(image_norm):
                return {
                    'error': 'no_leaves',
                    'message': 'No significant leaf area detected in the image'
                }
            
            # Segment leaves
            masks = self.segmenter.segment((image_norm * 255).astype(np.uint8))
            
            if masks is None or masks.size == 0:
                return {
                    'error': 'no_leaves',
                    'message': 'No individual leaves could be identified'
                }
            
            # Store masks for later use
            self.segmenter.masks = masks
            
            # If all_leaves is False (default), only process the first leaf
            if not all_leaves and leaf_index is None:
                leaf_index = 0
            
            # If leaf_index is specified, only process that leaf
            if leaf_index is not None:
                if leaf_index < 0 or leaf_index >= len(masks):
                    return {
                        'error': 'invalid_leaf',
                        'message': f'Leaf index {leaf_index} is out of range (0-{len(masks)-1})'
                    }
                masks = [masks[leaf_index]]
                
            # Predict for each leaf
            results = []
            for i, mask in enumerate(masks):
                leaf_img = self.segmenter.extract_leaf(image_norm, mask)
                # Ensure the image is the correct size for the model
                leaf_img = cv2.resize(leaf_img, self.img_size)
                pred = self.model.predict(np.expand_dims(leaf_img, 0), verbose=0)[0]
                
                # Get the predicted class and confidence
                pred_class_idx = np.argmax(pred)
                pred_class = self.class_names[pred_class_idx]
                confidence = float(pred[pred_class_idx])
                
                # Apply confidence thresholds
                if pred_class == 'healthy' and confidence < self.confidence_thresholds['healthy']:
                    # If healthy confidence is low, check other classes
                    powdery_conf = float(pred[1])  # Index 1 is powdery
                    rust_conf = float(pred[2])     # Index 2 is rust
                    
                    if powdery_conf > self.confidence_thresholds['powdery']:
                        pred_class = 'powdery'
                        confidence = powdery_conf
                    elif rust_conf > self.confidence_thresholds['rust']:
                        pred_class = 'rust'
                        confidence = rust_conf
                
                result = {
                    'leaf_index': i,
                    'class': pred_class,
                    'confidence': confidence,
                    'probabilities': {
                        cls: float(prob) for cls, prob in zip(self.class_names, pred)
                    }
                }
                
                results.append(result)
                
            return self._aggregate_results(results)
        except Exception as e:
            return {'error': 'processing_error', 'message': str(e)}
    
    def _aggregate_results(self, results):
        try:
            if not results:
                return {'error': 'no_results', 'message': 'No prediction results available'}
                
            return {
                'primary_class': results[0]['class'],
                'confidence': results[0]['confidence'],
                'per_leaf': results,
                'leaf_count': len(results)
            }
        except Exception as e:
            return {'error': 'aggregation_error', 'message': str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plant Disease Prediction Tool')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--leaf', type=int, help='Index of specific leaf to analyze (0-based)')
    parser.add_argument('--all', action='store_true', help='Analyze all leaves in the image')
    parser.add_argument('--explain', action='store_true', help='Show detailed explanations')
    args = parser.parse_args()
    
    try:
        predictor = PlantDiseasePredictor('models/best_model.keras')
        result = predictor.predict(args.image_path, leaf_index=args.leaf, explain=args.explain, all_leaves=args.all)
        
        if 'error' in result:
            if result['error'] == 'no_leaves':
                print("\nResult: No leaves detected in the image")
                print(f"Details: {result['message']}")
            elif result['error'] == 'invalid_leaf':
                print("\nError: Invalid leaf index")
                print(f"Details: {result['message']}")
            else:
                print(f"\nError: {result.get('message', 'Unknown error')}")
        else:
            print("\nPrediction Results:")
            print(f"Primary Classification: {result['primary_class']}")
            print(f"Overall Confidence: {result['confidence']:.2f}")
            
            if args.all:
                print(f"Number of leaves analyzed: {result['leaf_count']}")
                print("\nPer-leaf Analysis:")
                for leaf in result['per_leaf']:
                    print(f"\nLeaf {leaf['leaf_index'] + 1}:")
                    print(f"  Class: {leaf['class']}")
                    print(f"  Confidence: {leaf['confidence']:.2f}")
                    if args.explain:
                        print("  Probabilities:")
                        for cls, prob in leaf['probabilities'].items():
                            print(f"    {cls}: {prob:.2f}")
            else:
                leaf = result['per_leaf'][0]
                print(f"\nLeaf Analysis:")
                print(f"  Class: {leaf['class']}")
                print(f"  Confidence: {leaf['confidence']:.2f}")
                if args.explain:
                    print("  Probabilities:")
                    for cls, prob in leaf['probabilities'].items():
                        print(f"    {cls}: {prob:.2f}")
    except Exception as e:
        print(f"Fatal error: {str(e)}")