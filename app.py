import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

# print(cv2.__version__)
# print(cv2.__file__)
# print(libGL.__file__)

def print_title():
    st.title(':round_pushpin: Handwritten Digit Classifier')
    st.markdown("This model uses **Tensorflow** to create a **Convolutional Neural Network (CNN)**, with data taken from a classic MNIST dataset. The digits are randomly augmented (i.e., rotated, shifted, sheared, zoomed, and pixelated) to create more accurate results to a real handwriting use case.")

def print_credits():
    st.write("Dataset from: MNIST Handwritten Digit Dataset, with 60,000 digits for training and 10,000 digits for testing. Digits 0-9.")
    
def show_canvas_opts():
    # drawing_mode = st.sidebar.selectbox(
    # drawing_mode = st.selectbox(
    #     "Drawing tool:", ("freedraw", "transform")
    # )
    expander1 = st.expander(label=':gear: Canvas Options')
    with expander1:
        col1, col2 = st.columns(2)
        with col1:
            drawing_mode = st.selectbox(
                ":lower_left_paintbrush: Drawing Options:",
                ("freedraw", "transform"),
                index=0,  # Set the default index to 0 (freedraw)
                key="drawing_mode",  # Provide a unique key to the widget to ensure it updates correctly
            )
        with col2:
            stroke_width = st.slider(":straight_ruler: Stroke width: ", 1, 25, 14)
        return drawing_mode, stroke_width

def draw_canvas(drawing_mode, stroke_width):
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_color="#EFEFEF", # white-gray
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        # point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )
    return canvas_result

def open_img(drawn_image):
    # Load the image
    img = Image.open(drawn_image)

    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))

    # Convert the image to grayscale
    img = img.convert('L')

    # Invert colors (black to white, white to black)
    img = Image.eval(img, lambda x: 255 - x)

    # Convert the image data to a numpy array and normalize
    img_array = np.array(img) / 255.0

    # Add an extra dimension for the batch size and another for the channel
    img_array = img_array.reshape(1, 28, 28, 1)
    # st.image(img_array)
    return img_array

def model_predict(model, img_array):
    prediction = model.predict(img_array)
    return prediction    

def plot_prob(prediction):
    # Squeeze to remove single-dimensional entries from the shape of an array.
    prediction = np.squeeze(prediction)
    # sns.set(style="whitegrid")
    sns.set(style="whitegrid", rc={"grid.linewidth": 0.3})

    # Create a bar plot
    plt.figure(figsize=(5, 1))
    # plt.bar(range(10), prediction, color='blue')
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sns.barplot(x=digits, y=prediction*100, palette=sns.color_palette("mako"))
    plt.xlabel('Digits', fontsize=6)
    plt.ylabel('Probability (%)', fontsize=6)
    plt.title('Predicted Probabilities of Digits', fontsize=8)
    # Set font size for x-axis tick labels
    plt.xticks(range(10), fontsize=6)

    # Set font size for y-axis tick labels
    plt.yticks(np.arange(0, 101, 20), fontsize=5)

    # Remove the top and right spines/borders
    sns.despine()
    
    # Display the plot using st.pyplot()
    st.pyplot(plt)

def get_info():
    st.markdown("## :book: About This Model")
    st.write("This model uses a convolutional neural network to categorize handwritten digits.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Below is the confusion matrix for this model.")
        st.image("confusion_matrix.png")
    with col2:
        # st.write(model.summary())
        st.markdown("#### Below is the CNC layer diagram for this model.")
        st.image("cnc-digitrecognizer.png")

def main():
    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide")


    print_title()
    st.markdown("""---""")

    drawing_mode, stroke_width = show_canvas_opts()

    # Specify canvas parameters in application
    col1, col2 = st.columns([1, 2.5])
    with col1:
        canvas_result = draw_canvas(drawing_mode, stroke_width)

    # # Add a button to submit the drawing
    if st.button(":mag: Check Number"):
        if canvas_result.image_data is not None:
            # img = cv2.imwrite(f"drawn_image.png",  canvas_result.image_data)
            img = Image.fromarray(canvas_result.image_data)
            # Save the image to a file
            img.save("drawn_image.png")

            img_array = open_img("drawn_image.png")

            # model = keras.models.load_model("digits_recognition_cnn.h5")
            # prediction = model.predict(img_array)
            model = keras.models.load_model("digits_recognition_cnn.h5")
            # model = keras.models.load_model("model.keras")
            prediction = model_predict(model, img_array)

            predicted_digit = np.argmax(prediction)

            with col2:
                # col1a, col2a = st.columns([3, 1])
                # with col1a:
                    st.markdown(f"### :clipboard: Predicted Number: **`{predicted_digit}`**")
                    plot_prob(prediction)
                # with col2a:


        else:
            st.warning("Please draw something before submitting.")
    st.markdown("""---""")
    get_info()
    st.markdown("""---""")
    print_credits()

if __name__ == "__main__":
    main()
