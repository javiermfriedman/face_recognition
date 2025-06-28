import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam # The Adam optimizer
from tensorflow.keras.losses import BinaryCrossentropy # The correct loss for binary classification with sigmoid output
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def build_cnn_model(input_shape=(180, 180, 1)):
    """
    Builds a Convolutional Neural Network (CNN) model for binary image classification.

    Args:
        input_shape (tuple): The expected shape of each input image.                

    Returns:
        tf.keras.Model: The uncompiled CNN model. 
    """
    # Create Sequential model, layers are stacked one after another.
    model = Sequential([

        
        # --- First Convolutional Block ---
        # Conv2D: This layer learns different features (like edges or textures) from the image.
        #   32: We use 32 different filters (feature detectors) in this layer.
        #   (3, 3): Each filter looks at a 3x3 pixel area of the image.
        #   'relu': This activation function helps the network learn complex patterns.
        #   input_shape: Tells the model the exact size of the input images 
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),

        # MaxPooling2D: This layer shrinks the image size, keeping the most important information.
        #   (2, 2): It takes the biggest value from every 2x2 group of pixels, reducing the size by half.
        MaxPooling2D((2, 2)),

        # Dropout: Randomly turns off 25% of neurons during training.
        #   This helps prevent the model from memorizing the training data too much (overfitting).
        Dropout(0.25),


        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten: This layer converts the 3D output into a single long 1D list of numbers.
        Flatten(),

        # Dense: standard layer takes in 256 features from activation vector
        Dense(256, activation='relu'),
        Dropout(0.5), 

        # Dense: The output layer that gives the prediction.    
        Dense(1, activation='sigmoid')
    ])

    return model

def compile_and_train_model(model, X, y, epochs=40, batch_size=32, validation_split=0.2):
    print("[INFO] Compiling the model...")
    #   optimizer: Adam is a popular choice for efficient learning.
    #   loss: BinaryCrossentropy because we using binary classifivation 
    #         'from_logits=False' (default) means the model's output is already a probability.
    #   metrics: We want to track 'accuracy' during training.
    model.compile(
        loss=BinaryCrossentropy(from_logits=False), # Correct loss for sigmoid output
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Print a summary of the model's structure. This is very helpful!
    print("\n--- Model Summary ---")
    model.summary()
    print("\n[INFO] Model compiled successfully. Starting training...")

    # Split the data into training and validation sets make it random
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

    # Train the model!
    #   X_train, y_train: The data used for learning.
    #   epochs: How many times the model will see the entire training dataset.
    #   batch_size: How many samples are processed at once before the model updates its weights.
    #   validation_data: The data used to check performance on unseen examples during training.
    #   verbose=1: Shows a progress bar and training metrics per epoch.
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    print("[INFO] Model training completed.")
    return history

def plot_loss_tf(history):
    """
    Plots the training and validation loss from the model's training history.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()