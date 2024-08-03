                                    Digit Classification Application
## Overview
This application classifies handwritten digits (0-9) from uploaded images using a trained machine learning model. It utilizes a Convolutional Neural Network (CNN) trained on the MNIST dataset. The application is built with Streamlit for the web interface and TensorFlow/Keras for the machine learning model. Visualization of prediction probabilities is provided using Matplotlib and Seaborn.

## Features
Upload Image: Allows users to upload an image of a digit.
Predict: Classifies the uploaded image and displays the predicted digit along with a bar plot of prediction probabilities.
About: Provides information about the application and the technologies used.

## How to Use
Select 'Upload Image' from the sidebar.
Upload an image of a digit (0-9). The image should be in JPEG or PNG format.
Click the 'Predict' button to see the classification result and prediction probabilities.

## Model Information
Model Type: Convolutional Neural Network (CNN)
Dataset: MNIST dataset

## Technologies Used
Streamlit: For the web application interface
TensorFlow/Keras: For the machine learning model
Matplotlib: For plotting prediction probabilities
Seaborn: For improved visualization

## Installation and Setup

1.Clone the repository: 
git clone <repository-url>
cd <repository-directory>

2.Install the required Python packages:
pip install streamlit tensorflow pillow matplotlib seaborn numpy

3.Place your trained model (digit_classification.h5) in the ./model/ directory.
Run the Streamlit application:
streamlit run app.py

## Files in the Repository
app.py: The main Streamlit application script.
model/digit_classification.h5: The pre-trained CNN model.
README.md: This file.

## Acknowledgements
The model is trained on the MNIST dataset, a large database of handwritten digits. Thanks to the creators of Streamlit, TensorFlow/Keras, Matplotlib, and Seaborn for their amazing tools.