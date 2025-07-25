import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.preprocessing.image import img_to_array

# Load model and labels
model = tf.keras.models.load_model('flower_model.h5', compile=False)
with open('class_names.json') as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

def preprocess_image(image):
    image = image.resize((224, 224)).convert('RGB')
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

st.set_page_config(page_title="Indian Flower Classifier")
st.title("\U0001F33C Indian Flower Classifier")
st.write("Upload a flower image and get the predicted class.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        img_tensor = preprocess_image(image)
        predictions = model.predict(img_tensor)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    st.success(f"\U0001F338 Predicted: **{predicted_class}** ({confidence:.2f}%)")

