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
            {"urls": "stun:stun.services.mozilla.com"}
        ]
    }
)

# Video Transformer Class for Real-Time Processing
class EmotionDetection(VideoTransformerBase):
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.face_cascade.empty():
                st.error("Failed to load Haar Cascade classifier")
        except Exception as e:
            st.error(f"Error initializing classifier: {str(e)}")

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(img_gray, 1.3, 5)

            # Debugging: Show frame dimensions and face count
            st.write(f"Frame shape: {img.shape}, Faces detected: {len(faces)}")

            if len(faces) == 0:
                cv2.putText(img, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Simplified label (skip DeepFace for now to test)
                    label = "Unknown - Processing"
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Uncomment below to re-enable DeepFace (once basic detection works)
                    # face_roi = img[y:y + h, x:x + w]
                    # try:
                    #     result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    #     emotion = result[0]['dominant_emotion']
                    # except Exception as e:
                    #     emotion = f"Error: {str(e)}"
                    # label = f"Unknown - {emotion}"
                    # cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return img
        except Exception as e:
            st.error(f"Transform error: {str(e)}")
            return frame

# Streamlit App
def main():
    st.title("Real-Time Emotion Detection")
    st.write("This app detects faces and will identify emotions using your webcam.")

    # Start webcam streaming with error handling
    try:
        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=EmotionDetection,
            async_processing=True,
        )
    except Exception as e:
        st.error(f"WebRTC Error: {str(e)}")

if __name__ == "__main__":
    main()
