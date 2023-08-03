import streamlit as st
from PIL import Image
import streamlit-drawable-canvas as st_canvas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
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

    # Create a canvas to draw on
    canvas_result = st_canvas(
        fill_color="rgb(200, 200, 200)",  # Background color of the canvas
        stroke_width=5,  # Width of the stroke while drawing
        stroke_color="rgb(0, 0, 0)",  # Color of the stroke
        update_streamlit=True,
        height=200  # Height of the canvas in pixels
    )

    # Display the image drawn on the canvas
    if canvas_result.image_data is not None:
        st.write("You drew this:")
        st.image(canvas_result.image_data)

    # Add a button to submit the drawing
    if st.button("Submit Drawing"):
        if canvas_result.image_data is not None:
            # Save the drawing as an image file (optional)
            image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGB')
            image.save("drawn_image.png")
            img_array = open_img("drawn_image.png")

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
            st.pyplot()
            
        else:
            st.warning("Please draw something before submitting.")

if __name__ == "__main__":
    main()
