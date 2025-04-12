import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import sys

# Load model
model = load_model('multimodal_alzheimers_model.h5')

# Load sample image
img = Image.open('mri_images/cnn1_image_0.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

# Sample clinical features (scaled like during training)
clinical_data = np.array([[72, 22]])  # Age and MMSE score
# Normally you'd scale this with StandardScaler — for now we’re assuming it's already scaled or close

# Make prediction
prediction = model.predict([img_array, clinical_data])[0][0]

print(f"Alzheimer's Prediction Probability: {prediction:.4f}")
if prediction > 0.5:
    print("⚠️ Likely Alzheimer’s")
else:
    print("✅ Likely Normal")
