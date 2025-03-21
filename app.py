import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
from deepface import DeepFace

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video Transformer Class for Real-Time Processing
class EmotionDetection(VideoTransformerBase):
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

            # Display name (fixed as "Unknown") and emotion
            label = f"Unknown - {emotion}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

# Streamlit App
def main():
    st.title("Real-Time Emotion Detection")
    st.write("This app detects faces and identifies emotions using your webcam.")

    # Start webcam streaming
    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,  # Use enum instead of string
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=EmotionDetection,
    )

if __name__ == "__main__":
    main()
