import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import tempfile
import math
import time
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(page_title="Surakshit Sadak AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #00ffcc; font-family: 'Courier New'; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚦 Surakshit Sadak AI: Full-Stack Dashboard")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Loading on Apple Silicon MPS
    main_model = YOLO("yolov8n.pt")
    helmet_model = YOLO("surakshit_100_epochs.pt")
    return main_model, helmet_model

traffic_model, helmet_model = load_models()
tracker = sv.ByteTrack()

# --- SIDEBAR CONFIG ---
st.sidebar.header("🕹️ Control Panel")
source_mode = st.sidebar.radio("Input Source:", ["Upload Video File", "Live System Camera"])
conf_score = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.25)
show_metrics = st.sidebar.checkbox("Show Performance Metrics", value=True)

# --- PERSPECTIVE MATH ---
SOURCE_POLYGON = np.array([[307, 377], [864, 429], [1222, 679], [54, 673]], dtype=np.float32)
TARGET_POLYGON = np.array([[0, 0], [12, 0], [12, 30], [0, 30]], dtype=np.float32)
warp_matrix = cv2.getPerspectiveTransform(SOURCE_POLYGON, TARGET_POLYGON)

# --- INITIALIZE STATE ---
vehicle_history = {}
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.5)

# --- SOURCE SELECTION ---
video_source = None
if source_mode == "Upload Video File":
    uploaded_file = st.file_uploader("Upload MP4 Traffic Feed", type=["mp4", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_source = tfile.name
else:
    if st.sidebar.button("Launch Camera"):
        video_source = 0

# --- MAIN PROCESSING LOOP ---
if video_source is not None:
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    st_frame = st.empty() # The video display placeholder
    
    # Optional Metrics Dashboard
    if show_metrics:
        col1, col2, col3 = st.columns(3)
        m1 = col1.metric("Status", "Processing")
        m2 = col2.metric("Inference", "Apple M4 (MPS)")
        m3 = col3.metric("Violation Engine", "Active")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run Detection
        results = traffic_model(frame, device="mps", conf=conf_score, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])] # Cars, Motorcycles, etc.
        detections = tracker.update_with_detections(detections)

        labels = []
        for i in range(len(detections)):
            idx = detections.tracker_id[i]
            cls = detections.class_id[i]
            x1, y1, x2, y2 = detections.xyxy[i]
            
            # Speed Calculation
            bc_x, bc_y = (x1 + x2) / 2, y2
            px = np.array([[[bc_x, bc_y]]], dtype=np.float32)
            rx, ry = cv2.perspectiveTransform(px, warp_matrix)[0][0]
            
            if idx not in vehicle_history: vehicle_history[idx] = []
            vehicle_history[idx].append((rx, ry))
            if len(vehicle_history[idx]) > 5: vehicle_history[idx].pop(0)

            label = f"ID:{idx}"
            if len(vehicle_history[idx]) == 5:
                dist = math.sqrt((vehicle_history[idx][-1][0] - vehicle_history[idx][0][0])**2 + 
                                 (vehicle_history[idx][-1][1] - vehicle_history[idx][0][1])**2)
                speed = (dist / (5/fps)) * 3.6
                label += f" {int(speed)}km/h"
                
                # Wrong Way Detection
                if (vehicle_history[idx][-1][1] - vehicle_history[idx][0][1]) < -0.5:
                    label += " [WRONG WAY]"

            # Helmet Cascade (Motorcycles)
            if cls == 3:
                crop = frame[max(0,int(y1)):int(y2), max(0,int(x1)):int(x2)]
                if crop.size > 0:
                    h_res = helmet_model(crop, device="mps", verbose=False)[0]
                    if len(h_res.boxes) > 0:
                        h_cls = helmet_model.names[int(h_res.boxes[0].cls[0])]
                        label += f" [{h_cls.upper()}]"

            labels.append(label)

        # 2. Annotate
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # 3. Stream to Streamlit
        st_frame.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    st.success("Execution Finished.")