import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps

# Load the trained model
model = load_model('./model/digit_classification.h5')

# Function to preprocess image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = img_to_array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    image /= 255
    return image

# Streamlit app
st.set_page_config(page_title="Digit Classification")
st.title("✏️ Digit Classification Application")

# Sidebar for additional options
st.sidebar.title("Menu")
option = st.sidebar.selectbox("Choose an option", ["Upload Image", "About"])

if option == "Upload Image":
    st.sidebar.write("Upload an image of a digit (0-9).")
    # File uploader for image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # Predict button
        if st.button('Predict'):
            st.write("Classifying...")

            # Progress bar
            with st.spinner("Processing..."):
                # Preprocess the image
                preprocessed_image = preprocess_image(image)

                # Make a prediction
                prediction = model.predict(preprocessed_image)
                predicted_class = np.argmax(prediction, axis=1)[0]
                prediction_probabilities = prediction[0]

                st.write(f"Predicted Class: {predicted_class}")

                # Plot the prediction probabilities
                fig, ax = plt.subplots()
                sns.barplot(x=list(range(10)), y=prediction_probabilities, ax=ax, color='red')
                ax.set_xlabel('Digit Class')
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities')

                st.pyplot(fig)

else:
    # About page
    st.write("# About This App")
    st.write("""
    This Digit Classification app uses a trained machine learning model to predict 
    digits from uploaded images. This can classify digits from 0 to 9.

    ## How to use:
    1. Select 'Upload Image' from the sidebar.
    2. Upload an image of a digit (0-9).
    3. The input should be a grayscale image of size 28x28 pixels.
    3. Click the 'Predict' button to see the classification result.

    ## Model Information:
    - The model used is a Convolutional Neural Network (CNN) trained on the MNIST dataset.

    ## Technologies Used:
    - Streamlit for the web application
    - TensorFlow/Keras for the machine learning model
    - Matplotlib and Seaborn for visualization """)