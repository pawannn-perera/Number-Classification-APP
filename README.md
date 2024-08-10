
# Digit Classification Application

## Overview
This application is designed to classify handwritten digits (0-9) from uploaded images using a pre-trained Convolutional Neural Network (CNN). Built with a user-friendly interface using Streamlit, this tool leverages TensorFlow/Keras for machine learning and provides insightful visualizations using Matplotlib and Seaborn. 

## Features
- **Upload Image**: Upload an image of a digit (0-9) in JPEG or PNG format.
- **Predict**: Instantly classify the uploaded image and view the predicted digit along with a bar plot showing prediction probabilities.
- **About**: Learn more about the application, including the technologies and methodologies used.

## How to Use
1. **Upload an Image**: Use the 'Upload Image' option in the sidebar to upload a digit image.
2. **Predict**: Click on the 'Predict' button to classify the digit and view the prediction results.
3. **Visualize**: See the prediction probabilities visualized in a clear, easy-to-understand bar plot.

## Model Information
- **Model Type**: Convolutional Neural Network (CNN)
- **Dataset**: Trained on the MNIST dataset, a well-known dataset of handwritten digits.

## Technologies Used
- **Streamlit**: Simplifies the creation of the web application interface.
- **TensorFlow/Keras**: Powers the underlying machine learning model.
- **Matplotlib**: Used for plotting the prediction probabilities.
- **Seaborn**: Enhances the visualization with additional styling.

## Installation and Setup

1. **Clone the Repository**: 
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install streamlit tensorflow pillow matplotlib seaborn numpy
   ```

3. **Place the Trained Model**: 
   Save your pre-trained model as `digit_classification.h5` in the `./model/` directory.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Files in the Repository
- **app.py**: The main script for running the Streamlit application.
- **model/digit_classification.h5**: The trained CNN model file.
- **README.md**: Documentation and instructions for the application.

## Acknowledgements
This project is built upon the MNIST dataset, a comprehensive collection of handwritten digits. Special thanks to the developers of Streamlit, TensorFlow/Keras, Matplotlib, and Seaborn for their incredible tools that made this application possible.
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
