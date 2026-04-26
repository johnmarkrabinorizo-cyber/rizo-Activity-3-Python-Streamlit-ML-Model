import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import os
from datetime import datetime
from collections import defaultdict

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="Live Object Detection & Tracing",
    page_icon="🎥",
    layout="wide"
)

# ----------------------
# Load Model
# ----------------------
@st.cache_resource
def load_model():
    # Load YOLOv8 nano model (lightweight and fast for real-time use)
    return YOLO("yolov8n.pt")

model = load_model()

# ----------------------
# Create Folder for Saved Frames
# ----------------------
if not os.path.exists("detected_frames"):
    os.makedirs("detected_frames")

# ----------------------
# Session State Initialization
# ----------------------
if "object_counts" not in st.session_state:
    st.session_state.object_counts = defaultdict(int)
if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False
# Set which objects will trigger an alert
ALERT_OBJECTS = ["cell phone", "knife", "bottle"]  

# ----------------------
# App Title and Description
# ----------------------
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify, track, and count them in real-time.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    save_frames = st.checkbox("Save Detected Frames Automatically", value=False)
    st.markdown("---")
    st.subheader("🔔 Alert Settings")
    st.write(f"Alert will trigger when these objects are detected: {', '.join(ALERT_OBJECTS)}")

# ----------------------
# Video Frame Processing
# ----------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    detected_classes = []

    # Run object detection and tracking
    results = model.track(
        img,
        persist=True,
        conf=confidence,
        verbose=False
    )

    # Get detected object details
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Reset counts before updating
        current_counts = defaultdict(int)
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            current_counts[class_name] += 1
            detected_classes.append(class_name)

        # Update session state counts
        st.session_state.object_counts = current_counts

        # Check for alert objects
        if any(obj in detected_classes for obj in ALERT_OBJECTS):
            st.session_state.alert_triggered = True
        else:
            st.session_state.alert_triggered = False

        # Save frame if enabled
        if save_frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = f"detected_frames/frame_{timestamp}.jpg"
            annotated_img = results[0].plot()
            cv2.imwrite(save_path, annotated_img)

    # Draw annotations on frame
    annotated_frame = results[0].plot() if len(results) > 0 else img

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ----------------------
# Display Stream and Results
# ----------------------
col1, col2 = st.columns([3, 1])

with col1:
    # Start webcam stream
    webrtc_streamer(
        key="object-detection",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Object Count")
    if st.session_state.object_counts:
        for obj, count in st.session_state.object_counts.items():
            st.write(f"• {obj}: {count}")
    else:
        st.write("No objects detected yet.")

    st.markdown("---")

    st.subheader("⚠️ Status Alert")
    if st.session_state.alert_triggered:
        st.error("⚠️ Alert: Restricted or specified object detected!")
    else:
        st.success("✅ No alert-triggering objects detected.")

# ----------------------
# Additional Information
# ----------------------
st.markdown("---")
st.info("💡 **Note**: All saved frames are stored in the `detected_frames` folder in your working directory.")