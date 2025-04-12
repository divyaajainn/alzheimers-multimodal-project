# train.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import create_multimodal_model
from sklearn.model_selection import train_test_split

def load_data():
    """Load preprocessed data saved as NumPy arrays."""
    images = np.load('images.npy')
    clinical_features = np.load('clinical_features.npy')
    labels = np.load('labels.npy')
    return images, clinical_features, labels

def split_data(images, clinical_features, labels, test_size=0.2, random_state=42):
    """Split the dataset into training and validation sets."""
    train_images, val_images, train_clinical, val_clinical, train_labels, val_labels = train_test_split(
        images, clinical_features, labels, test_size=test_size, random_state=random_state)
    return train_images, val_images, train_clinical, val_clinical, train_labels, val_labels

def plot_history(history):
    """Plot training and validation accuracy and loss over epochs."""
    plt.figure(figsize=(12, 4))
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Load preprocessed data
    images, clinical_features, labels = load_data()
    
    # Split the data into training and validation sets
    train_images, val_images, train_clinical, val_clinical, train_labels, val_labels = split_data(
        images, clinical_features, labels)
    
    # Determine the number of clinical features
    clinical_dim = train_clinical.shape[1]
    
    # Build and compile the multimodal model
    model = create_multimodal_model(image_shape=(224, 224, 3), clinical_dim=clinical_dim)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        [train_images, train_clinical],
        train_labels,
        epochs=25,
        batch_size=32,
        validation_data=([val_images, val_clinical], val_labels)
    )
    
    # Evaluate model performance on validation data
    loss, accuracy = model.evaluate([val_images, val_clinical], val_labels)
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')
    
    # Plot training history
    plot_history(history)
    
    # Save the trained model
    model.save('multimodal_alzheimers_model.h5')
    print("Model saved as 'multimodal_alzheimers_model.h5'.")

if __name__ == '__main__':
    main()
