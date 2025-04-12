# model.py

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model

def create_image_branch(input_shape=(224, 224, 3)):
    """Creates the CNN branch for processing MRI images."""
    image_input = Input(shape=input_shape, name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    image_output = layers.Dense(128, activation='relu')(x)
    return Model(inputs=image_input, outputs=image_output, name='ImageBranch')

def create_clinical_branch(input_dim):
    """Creates the branch for processing clinical (tabular) data."""
    clinical_input = Input(shape=(input_dim,), name='clinical_input')
    y = layers.Dense(64, activation='relu')(clinical_input)
    y = layers.Dense(32, activation='relu')(y)
    return Model(inputs=clinical_input, outputs=y, name='ClinicalBranch')

def create_multimodal_model(image_shape=(224, 224, 3), clinical_dim=3):
    """Combines the image and clinical branches, adds a fusion layer, and outputs a binary prediction."""
    image_branch = create_image_branch(input_shape=image_shape)
    clinical_branch = create_clinical_branch(input_dim=clinical_dim)
    
    # Fuse the outputs
    combined = layers.concatenate([image_branch.output, clinical_branch.output], name='fusion_layer')
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(32, activation='relu')(z)
    output = layers.Dense(1, activation='sigmoid', name='output')(z)
    
    model = Model(inputs=[image_branch.input, clinical_branch.input], outputs=output, name='Multimodal_Alzheimers_Model')
    return model

if __name__ == '__main__':
    # A quick test: Change clinical_dim as needed depending on your features
    test_model = create_multimodal_model(image_shape=(224, 224, 3), clinical_dim=3)
    test_model.summary()
