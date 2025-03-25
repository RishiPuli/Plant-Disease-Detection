import tensorflow as tf
import numpy as np
import cv2
import os
from utils.dataset import PlantDataset
import pandas as pd

class PlantDiseasePredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['healthy', 'powdery', 'rust']
        self.img_size = (256, 256)
    
    def predict_image(self, img_path):
        """Predict disease from single image"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Could not read image (possibly corrupt or unsupported format)")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            pred = self.model.predict(img, verbose=0)
            class_idx = np.argmax(pred)
            confidence = np.max(pred)
            
            return {
                'class': self.class_names[class_idx],
                'confidence': float(confidence),
                'probabilities': {
                    cls: float(prob) for cls, prob in zip(self.class_names, pred[0])
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed for {img_path}: {str(e)}")

def batch_predict(image_dir, output_csv='predictions.csv'):
    """Predict for all images in a directory"""
    try:
        predictor = PlantDiseasePredictor('models/best_model.keras')
        results = []
        
        for img_name in os.listdir(image_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(image_dir, img_name)
                    result = predictor.predict_image(img_path)
                    results.append({
                        'image': img_name,
                        **result
                    })
                except Exception as e:
                    print(f"Skipping {img_name}: {str(e)}")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved predictions for {len(results)} images to {output_csv}")
        return df
        
    except Exception as e:
        print(f"Batch prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plant Disease Prediction')
    parser.add_argument('input_path', help='Path to image file or directory')
    parser.add_argument('--output', help='Output CSV path for batch prediction', default='predictions.csv')
    args = parser.parse_args()
    
    if os.path.isfile(args.input_path):
        # Single image prediction
        predictor = PlantDiseasePredictor('models/best_model.keras')
        result = predictor.predict_image(args.input_path)
        print("\nPrediction Result:")
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Class Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
            
    elif os.path.isdir(args.input_path):
        # Batch prediction
        batch_predict(args.input_path, args.output)
    else:
        print(f"Invalid path: {args.input_path}")