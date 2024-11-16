
# Number Classification APP

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
