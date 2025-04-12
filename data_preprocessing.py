# data_preprocessing.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Parameters
IMAGE_DIR = 'mri_images'
CLINICAL_CSV = 'clinical_data.csv'
TARGET_SIZE = (224, 224)

def load_and_preprocess_image(image_path, target_size=TARGET_SIZE):
    """Load an image, convert to RGB, resize, and normalize."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0,1]
    return image_array

def load_images(image_dir=IMAGE_DIR):
    """Load and preprocess all images from the directory.
       Returns a NumPy array of images and their corresponding IDs."""
    images = []
    image_ids = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            img_array = load_and_preprocess_image(image_path)
            images.append(img_array)
            # Use the filename (without extension) as the patient ID
            patient_id = os.path.splitext(filename)[0]
            image_ids.append(patient_id)
    return np.array(images), image_ids

def load_clinical_data(csv_file=CLINICAL_CSV):
    """Load clinical data from a CSV file and fill missing values for numeric columns."""
    df = pd.read_csv(csv_file)
    # Fill missing numeric values only (non-numeric columns are left untouched)
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
    return df


def align_data(images, image_ids, clinical_df):
    """
    Align images with clinical data based on patient_id.
    Assumes the clinical CSV has a 'patient_id' column and a 'label' column.
    Returns aligned images, clinical features, and labels.
    """
    # Filter clinical data to only include records with matching image IDs.
    aligned_clinical_df = clinical_df[clinical_df['patient_id'].isin(image_ids)].copy()
    clinical_df_indexed = aligned_clinical_df.set_index('patient_id')

    clinical_features = []
    valid_images = []
    valid_labels = []
    for patient_id, img in zip(image_ids, images):
        if patient_id in clinical_df_indexed.index:
            # Drop the 'label' column from the features
            row = clinical_df_indexed.loc[patient_id]
            features = row.drop(['label'], errors='ignore').values.astype(float)
            clinical_features.append(features)
            valid_images.append(img)
            valid_labels.append(row['label'])
    return np.array(valid_images), np.array(clinical_features), np.array(valid_labels)

def scale_clinical_data(clinical_features):
    """Scale clinical features using StandardScaler."""
    scaler = StandardScaler()
    clinical_features_scaled = scaler.fit_transform(clinical_features)
    return clinical_features_scaled, scaler

if __name__ == '__main__':
    images, image_ids = load_images()
    clinical_df = load_clinical_data()
    images_aligned, clinical_features, labels = align_data(images, image_ids, clinical_df)
    clinical_features_scaled, scaler = scale_clinical_data(clinical_features)
    
    # Save preprocessed data for training (as .npy files)
    np.save('images.npy', images_aligned)
    np.save('clinical_features.npy', clinical_features_scaled)
    np.save('labels.npy', labels)
    
    print("Preprocessing complete. Saved files: images.npy, clinical_features.npy, labels.npy.")
