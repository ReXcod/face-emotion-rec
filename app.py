import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
from deepface import DeepFace

# RTC Configuration for WebRTC with multiple STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
            {"urls": "stun:stun2.l.google.com:19302"},
            {"urls": "stun:stun.services.mozilla.com"}  # Additional fallback
        ]
    }
)

# Video Transformer Class for Real-Time Processing
class EmotionDetection(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(img_gray, 1.3, 5)

            # Debugging: Log if faces are detected
            if len(faces) > 0:
                st.write(f"Detected {len(faces)} faces")  # Temporary debug output

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Extract face ROI
                face_roi = img[y:y + h, x:x + w]
                
                # Detect emotion (optional: comment out to test performance)
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                except Exception as e:
                    emotion = f"Error: {str(e)}"

                # Display name and emotion
                label = f"Unknown - {emotion}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return img
        except Exception as e:
            st.error(f"Transform error: {str(e)}")  # Log any processing errors
            return frame  # Return unprocessed frame as fallback

# Streamlit App
def main():
    st.title("Real-Time Emotion Detection")
    st.write("This app detects faces and identifies emotions using your webcam.")

    # Start webcam streaming with error handling
    try:
        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=EmotionDetection,
            async_processing=True,  # Enable async to reduce lag
        )
    except Exception as e:
        st.error(f"WebRTC Error: {str(e)}")

if __name__ == "__main__":
    main()
