import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.dataset import PlantDataset
import matplotlib.pyplot as plt
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')  # Absolute path
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 30

print(f"Using data from: {DATA_DIR}")

try:
    # Initialize dataset
    dataset = PlantDataset(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    train_gen = dataset.get_generator('train', augment=True)
    val_gen = dataset.get_generator('val')
    
    print(f"Found {train_gen.samples} training images")
    print(f"Found {val_gen.samples} validation images")

    # Model architecture
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = True

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    os.makedirs('models', exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    # Training
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_steps=val_gen.samples // BATCH_SIZE
    )

    # Save training plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    
    plt.savefig('training_history.jpg')
    plt.show()

    # Add this at the end of train.py (before the plt.show())
    model.save('models/best_model.keras')
    print("Model explicitly saved to models/best_model.keras")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("\nTroubleshooting Tips:")
    print("1. Verify your data folder structure matches exactly")
    print("2. Check all class folders contain images")
    print(f"3. Confirm path exists: {DATA_DIR}")
    print("4. Image files should be .jpg")