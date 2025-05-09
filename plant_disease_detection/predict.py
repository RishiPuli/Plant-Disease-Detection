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
            self.img_size = (512, 512)
            # Threshold for considering an image as containing leaves
            self.leaf_threshold = 0.1  # 10% of image area must be green
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
    
    def predict(self, image_path, explain=False):
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
            
            if not masks:
                return {
                    'error': 'no_leaves',
                    'message': 'No individual leaves could be identified'
                }
            
            # Store masks for later use
            self.segmenter.masks = masks
                
            # Predict for each leaf
            results = []
            for mask in masks:
                leaf_img = self.segmenter.extract_leaf(image_norm, mask)
                pred = self.model.predict(np.expand_dims(leaf_img, 0), verbose=0)[0]
                
                result = {
                    'class': self.class_names[np.argmax(pred)],
                    'confidence': float(np.max(pred)),
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--explain', action='store_true', help='Show explanations')
    args = parser.parse_args()
    
    try:
        predictor = PlantDiseasePredictor('models/best_model.keras')
        result = predictor.predict(args.image_path, explain=args.explain)
        
        if 'error' in result:
            if result['error'] == 'no_leaves':
                print("\nResult: No leaves detected in the image")
                print(f"Details: {result['message']}")
            else:
                print(f"\nError: {result.get('message', 'Unknown error')}")
        else:
            print("\nPrediction Results:")
            print(f"Primary Classification: {result['primary_class']}")
            print(f"Overall Confidence: {result['confidence']:.2f}")
            print(f"Number of leaves detected: {result['leaf_count']}")
            
            if len(result['per_leaf']) > 1:
                print("\nPer-leaf Analysis:")
                for i, leaf in enumerate(result['per_leaf']):
                    print(f"\nLeaf {i+1}:")
                    print(f"  Class: {leaf['class']}")
                    print(f"  Confidence: {leaf['confidence']:.2f}")
                    print("  Probabilities:")
                    for cls, prob in leaf['probabilities'].items():
                        print(f"    {cls}: {prob:.2f}")
    except Exception as e:
        print(f"Fatal error: {str(e)}")