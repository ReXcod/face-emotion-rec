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

            st.write(f"Frame shape: {img.shape}, Faces detected: {len(faces)}")

            if len(faces) == 0:
                cv2.putText(img, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_roi = img[y:y + h, x:x + w]
                    try:
                        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        emotion = result[0]['dominant_emotion']
                    except Exception as e:
                        emotion = f"Error: {str(e)}"
                    label = f"Unknown - {emotion}"
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return img
        except Exception as e:
            st.error(f"Transform error: {str(e)}")
            return frame

# Streamlit App
def main():
    st.title("Real-Time Emotion Detection")
    st.write("This app detects faces and identifies emotions using your webcam.")

    # Status placeholder
    status = st.empty()

    # Refresh button
    if st.button("Refresh Webcam"):
        st.rerun()

    try:
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=EmotionDetection,
            # Removed async_processing to test stability
        )
        if webrtc_ctx.state.signaling:
            status.info("WebRTC signaling is active (device selected)")
        if webrtc_ctx.state.playing:
            status.success("Webcam is active and streaming")
        else:
            status.warning("Webcam is not streaming - check permissions or refresh")
    except Exception as e:
        status.error(f"WebRTC Error: {str(e)}")
        st.write("Troubleshooting: Ensure webcam permissions are granted, no other apps are using the camera, and try a different browser.")

if __name__ == "__main__":
    main()
