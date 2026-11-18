
# Handwritten Digit Prediction

A handwritten digit recognition project using
Convolutional Neural Network (CNN) and Vision Transformer (ViT).

## Features

- Handwritten digit classification (0-9)
- CNN deep learning model
- Vision Transformer with 16 image patches
- Shared stratified dataset splitting
- Accuracy and Macro F1 evaluation
- Confusion matrices
- Majority-class baseline
- Image preprocessing using Pillow
- Streamlit web application
- Save and reload trained model weights
- Compare CNN and ViT predictions

## Dataset

Uses sklearn.datasets.load_digits().

- 1797 images
- 8x8 grayscale images
- 10 digit classes
- Pixel values normalized from 0-16 to 0-1

## Installation

Install dependencies:

pip install -r requirements.txt

## Training

Run:

python train.py

This trains both models and saves their checkpoints
inside the models folder.

Evaluation results and graphs are saved inside results.

## Run Application

Run:

streamlit run app.py

## Models

1. CNN
2. Vision Transformer

Both models classify handwritten digits from 0 to 9.

## Technologies

- Python
- PyTorch
- Scikit-learn
- NumPy
- Pillow
- Matplotlib
- Seaborn
- Streamlit

## Author

ZARFSHAN ATTIQ KHAN

## Github Link

https://github.com/zarfshankhan/Handwritten-Digit-Prediction.git


## App View

https://zarfshankhan-handwritten-digit-prediction-app-rcp1z4.streamlit.app/