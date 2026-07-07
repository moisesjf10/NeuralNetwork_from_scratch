import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

from network import NeuralNetwork
from layers import Layer, FlattenLayer, PoolingLayer, ConvLayer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy
from optimizers import Adam


#Load model (in cache to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    net = NeuralNetwork(
        layers=[
            ConvLayer(1,8,3,padding=1),
            PoolingLayer(),
            ConvLayer(8,16,3,padding=1),
            PoolingLayer(),
            FlattenLayer(),
            Layer(16*7*7, 64, ReLU()),
            Layer(64, 10, Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )

    net.load_weights("models/cnn_mnist.pkl")
    return net

net = load_model()

#USER INTERFACE
st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the canvas below")

#divide the page into two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Canvas")
    canvas_result = st_canvas(
        fill_color="#000000",       # Fill color (black)
        stroke_width=20,            # Brush size
        stroke_color="#FFFFFF",     # white stroke color
        background_color="#000000", # background color (black)
        height=280,                 # canvas 280x280
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Neural Network Analysis")
    
    # if user has drawn something on the canvas
    if canvas_result.image_data is not None:
        # extract the image (comes in RGBA format 280x280)
        img_rgba = canvas_result.image_data
        
        # greyscale, resize, normalize and reshape the image to fit the network input
        img_gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
        
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # adjust the tensor shape for the network input (Batch=1, channels=1, height=28, width=28)
        input_tensor = img_normalized.reshape(1, 1, 28, 28)
        
        # If the image is not completely empty (black)
        if np.sum(input_tensor) > 0:
            #infer the probabilities of each digit using the neural network
            probs = net.forward(input_tensor)[0]
            
            predict = np.argmax(probs)
            trust = probs[predict] * 100
            
            st.markdown(f"### Predict: **{predict}**")
            st.progress(int(trust), text=f"Trust: {trust:.2f}%")
            
            # bar chart of probabilities for each digit
            st.bar_chart(probs)
            
            st.write("### Input Image")
            st.image(img_resized, width=100)
        else:
            st.info("Waiting for your drawing...")