import json

from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import math
import datetime
import platform
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_path = 'model.keras'
loaded_model = tf.keras.models.load_model(model_path)

def print_title():
    st.title('Handwritten Digit Classifier')
    st.write("This model uses Tensorflow to create a Convolutional Neural Network (CNN).")
    st.write("The digits are randomly augmented (rotated, shifted, sheared, zoomed, and pixelated) to create more accurate results to a real handwriting use case.")
    st.markdown("""---""")

def print_credits():
    st.write("Dataset from: MNIST Handwritten Digit Dataset, with 60,000 digits for training and 10,000 digits for testing. Digits 0-9.")
    
def open_img(drawn_image):
    # Load the image
    img = Image.open(drawn_image)
    st.image(img)

    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))

    # Convert the image to grayscale
    img = img.convert('L')

    # Convert the image data to a numpy array and normalize
    img_array = np.array(img) / 255.0

    # Add an extra dimension for the batch size and another for the channel
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def main():
    print_title()
    
     # Specify canvas parameters in application

    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "transform")
    )
    
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=150,
        width=150,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    # Display the image drawn on the canvas
    if canvas_result.image_data is not None:
        st.write("You drew this:")
        st.image(canvas_result.image_data)

    # # Add a button to submit the drawing
    if st.button("Submit Drawing"):
        if canvas_result.image_data is not None:            
            # Save the drawing as an image file (optional)
            # Assuming that canvas_result.image_data is a JSON string representing the canvas data
            canvas_data = json.loads(canvas_result.image_data)
            
            # Extract the image data as a numpy array
            image_data = np.array(canvas_data['pixels'], dtype=np.uint8)
            
            # Reshape the data to match the canvas dimensions (e.g., height x width x 3 for RGB image)
            height = canvas_data['height']
            width = canvas_data['width']
            image_data = image_data.reshape(height, width, 3)
            # image = Image.fromarray(canvas_result.image_data, 'RGB')
            image = Image.fromarray(image_data, 'RGB')

            # st.write(image)
            st.image(image)


            image.save("drawn_image.png")
            # image.save("drawn_image", "png")
            # st.image("drawn_img.png")
            img_array = open_img("drawn_image.png")

            model = keras.models.load_model("model.keras")

            # Get the prediction
            prediction = model.predict(img_array)

            # The output of the model is a 10-element vector with the probabilities for each digit.
            # Use argmax to get the digit with the highest probability.
            predicted_digit = np.argmax(prediction)
            print('The number is predicted to be: ', predicted_digit)

            # Squeeze to remove single-dimensional entries from the shape of an array.
            prediction = np.squeeze(prediction)
            
            # Create a bar plot
            plt.figure(figsize=(9, 3))
            plt.bar(range(10), prediction)
            plt.xlabel('Digits')
            plt.ylabel('Probabilities')
            plt.title('Predicted probabilities of Digits')
            plt.xticks(range(10))
            
            # Display the plot using st.pyplot()
            st.pyplot(plt)
            
        else:
            st.warning("Please draw something before submitting.")

if __name__ == "__main__":
    main()
