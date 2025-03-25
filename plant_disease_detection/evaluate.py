import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.dataset import PlantDataset
import os

def evaluate_model():
    try:
        # Load model
        if not os.path.exists('models/best_model.keras'):
            raise FileNotFoundError("Trained model not found. Please train first.")
            
        model = tf.keras.models.load_model('models/best_model.keras')
        
        # Load dataset
        dataset = PlantDataset('data')
        test_gen = dataset.get_test_generator()
        
        print(f"\nFound {test_gen.samples} test images")
        
        # Evaluation
        print("\nEvaluating model...")
        test_loss, test_acc = model.evaluate(test_gen)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Detailed metrics
        y_true = test_gen.classes
        y_pred = model.predict(test_gen)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=dataset.class_names))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=dataset.class_names,
                    yticklabels=dataset.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.show()
        
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model()