from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your Next.js app's URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_sift_features(img_path, max_keypoints=100, descriptor_size=128):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image for SIFT feature extraction.")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # Handle cases where descriptors are None
    if descriptors is None:
        descriptors = np.zeros((max_keypoints, descriptor_size))

    # Truncate or pad descriptors to ensure consistent size
    if descriptors.shape[0] > max_keypoints:
        descriptors = descriptors[:max_keypoints]
    elif descriptors.shape[0] < max_keypoints:
        padding = np.zeros((max_keypoints - descriptors.shape[0], descriptor_size))
        descriptors = np.vstack([descriptors, padding])

    return descriptors.flatten()

def preprocess_image_and_sift(img_path, target_size=(224, 224), scaler=None, max_keypoints=100, descriptor_size=128):
    # Extract SIFT features
    sift_features = extract_sift_features(img_path, max_keypoints=max_keypoints, descriptor_size=descriptor_size)

    # Normalize SIFT features if a scaler is provided
    if scaler is not None:
        sift_features = scaler.transform([sift_features])  # Wrap in a list for batch dimension
    sift_features = np.expand_dims(sift_features[0], axis=0)  # Ensure shape is (1, feature_size)

    # Preprocess the image for CNN
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array, sift_features


# Load the trained scaler and model
try:
    scaler = joblib.load('sift_scaler.pkl')
    model = load_model('glaucoma_model.h5')
except Exception as e:
    raise RuntimeError(f"Failed to load required resources: {str(e)}")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        temp_file_path = "uploaded_image.png"
        with open(temp_file_path, "wb") as f:
            f.write(await image.read())

        cnn_input, sift_input = preprocess_image_and_sift(temp_file_path, scaler=scaler)

        prediction = model.predict([cnn_input, sift_input])

        label_map = {"0": "Glaucoma Present", "1": "Glaucoma not Present"}
        threshold = float(prediction[0][0])
        predicted_label = label_map[str(int(threshold < 0.5))]

        os.remove(temp_file_path)

        return JSONResponse(
            content={
                "prediction": predicted_label,
                "threshold": f"{round(threshold * 100, 2)}"
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
