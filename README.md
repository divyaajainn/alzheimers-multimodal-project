# Multimodal Alzheimer's Detection (ML Project)

This project uses MRI brain images and clinical data (like age and MMSE score) to predict the likelihood of Alzheimer's using a multimodal deep learning model.

## Project Includes:
- MRI image preprocessing
- Clinical data alignment and scaling
- A dual-branch neural network (image + tabular input)
- Training and saving the model
- A prediction script for new inputs

## Requirements:
- Python 3.8+
- TensorFlow, NumPy, Pandas, Pillow, scikit-learn

To run:
```bash
python data_preprocessing.py
python train.py
