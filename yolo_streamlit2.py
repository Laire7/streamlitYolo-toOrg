import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Set up the sidebar with model selection
st.sidebar.title("YOLOv8 Model Selection")
model_choice = st.sidebar.radio("Choose YOLOv8 model:", ["yolov8n", "yolov8s", "yolov8m"])

# Load the selected model
@st.cache_resource
def load_model(model_choice):
    return YOLO(model_choice)

model = load_model(model_choice)

def predict(frame, iou=0.7, conf=0.25):
    results = model.predict(
        source=frame,
        device="0" if torch.cuda.is_available() else "cpu",
        iou=iou,
        conf=conf,
        verbose=False,
    )
    return results[0]  # Access the first result if running on a single image

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Read and convert the image to numpy array
    image = Image.open(uploaded_image)
    np_image = np.array(image)

    # Display original image
    st.image(image, caption="Original Image", use_column_width=True, clamp=True)

    # Run model prediction
    result = predict(np_image)

    # Draw red bounding boxes on the original image
    result_image = np_image.copy()
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get coordinates
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red box, thickness 1
    
    # Display prediction result
    st.image(result_image, caption="Prediction Result", use_column_width=True)
