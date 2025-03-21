import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
from deepface import DeepFace
import requests
import base64

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Predefined known face embeddings (loaded from GitHub-hosted images)
KNOWN_EMBEDDINGS = {}
KNOWN_NAMES = {}

# Function to load an image from a URL and get its embedding
def load_image_embedding_from_url(url):
    try:
        response = requests.get(url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        return embedding
    except:
        return None

# Initialize known faces from GitHub-hosted images
def initialize_known_faces():
    global KNOWN_EMBEDDINGS, KNOWN_NAMES
    known_face_urls = {
        "Person1": "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/known_faces/person1.jpg",
        "Person2": "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/known_faces/person2.jpg"
    }
    for name, url in known_face_urls.items():
        embedding = load_image_embedding_from_url(url)
        if embedding is not None:
            KNOWN_EMBEDDINGS[name] = embedding
            KNOWN_NAMES[name] = name  # Could be PRN instead

# Video Transformer Class for Real-Time Processing
class EmotionAndFaceRecognition(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img_gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face ROI
            face_roi = img[y:y + h, x:x + w]
            
            # Detect emotion
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except:
                emotion = "Unknown"

            # Simulate face recognition with embeddings
            try:
                current_embedding = DeepFace.represent(face_roi, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                name = "Unknown"
                min_dist = float("inf")
                for known_name, known_embedding in KNOWN_EMBEDDINGS.items():
                    dist = np.linalg.norm(np.array(current_embedding) - np.array(known_embedding))
                    if dist < min_dist and dist < 0.6:  # Threshold for similarity
                        min_dist = dist
                        name = KNOWN_NAMES[known_name]
            except:
                name = "Unknown"

            # Display name and emotion
            label = f"{name} - {emotion}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

# Streamlit App
def main():
    st.title("Real-Time Face Recognition and Emotion Detection")
    st.write("This app detects faces, recognizes them, and identifies emotions using your webcam.")

    # Initialize known faces
    if not KNOWN_EMBEDDINGS:
        with st.spinner("Loading known faces..."):
            initialize_known_faces()

    # Start webcam streaming
    webrtc_streamer(
        key="example",
        mode="sendrecv",
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=EmotionAndFaceRecognition,
    )

if __name__ == "__main__":
    main()
