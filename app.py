import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
from deepface import DeepFace
import face_recognition
import base64
import requests

# RTC Configuration for WebRTC (works online)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Predefined known faces (encoded remotely or hardcoded for simplicity)
# For online use, we'll simulate loading known faces from a GitHub-hosted image
KNOWN_FACES = {}
KNOWN_NAMES = {}

# Function to load an image from a URL and encode it
def load_image_from_url(url):
    response = requests.get(url)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    return encodings[0] if encodings else None

# Load known faces from GitHub-hosted images (replace with your GitHub raw URLs)
def initialize_known_faces():
    global KNOWN_FACES, KNOWN_NAMES
    known_face_urls = {
        "Person1": "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/known_faces/person1.jpg",
        "Person2": "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/known_faces/person2.jpg"
    }
    for name, url in known_face_urls.items():
        encoding = load_image_from_url(url)
        if encoding is not None:
            KNOWN_FACES[name] = encoding
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
        
        # Convert to RGB for face_recognition
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face ROI for emotion detection
            face_roi = img[y:y + h, x:x + w]
            try:
                # Detect emotion using DeepFace
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except:
                emotion = "Unknown"

            # Face recognition
            face_encoding = face_recognition.face_encodings(rgb_img, [(y, x + w, y + h, x)])
            name = "Unknown"
            if face_encoding:
                for known_name, known_encoding in KNOWN_FACES.items():
                    match = face_recognition.compare_faces([known_encoding], face_encoding[0])
                    if match[0]:
                        name = KNOWN_NAMES[known_name]
                        break

            # Display name and emotion
            label = f"{name} - {emotion}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

# Streamlit App
def main():
    st.title("Real-Time Face Recognition and Emotion Detection")
    st.write("This app detects faces, recognizes them, and identifies emotions using your webcam.")

    # Initialize known faces
    if not KNOWN_FACES:
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
