import cv2
import math
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import time
from datetime import datetime

# --- CREATE EVIDENCE FOLDER ---
if not os.path.exists("violations"):
    os.makedirs("violations")

print("Loading Surakshit Sadak AI Dual-Core System...")

# --- THE DUAL MODELS ---
# Brain 1: Default YOLOv8 for detecting vehicles and tracking movement
traffic_model = YOLO("yolov8n.pt")                 
# Brain 2: Your custom 100-epoch model for precise helmet detection
helmet_model = YOLO("surakshit_100_epochs.pt")     
tracker = sv.ByteTrack()

cap = cv2.VideoCapture("sample_traffic.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# 1. Perspective Math (Converting Pixels to Real-World Meters)
SOURCE_POLYGON = np.array([[307, 377], [864, 429], [1222, 679], [54, 673]], dtype=np.float32)
TARGET_POLYGON = np.array([[0, 0], [12, 0], [12, 30], [0, 30]], dtype=np.float32)
warp_matrix = cv2.getPerspectiveTransform(SOURCE_POLYGON, TARGET_POLYGON)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

# Analytics Tracking Memory
vehicle_position_history = {}
violators_caught = {} 

print("Starting Automated Enforcement Stream...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    annotated_frame = frame.copy()
    
    # --- NEW: LIVE CCTV TIMESTAMP OVERLAY ---
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, f"CAM-01 | {current_time}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw the yellow speed zone polygon
    pts = SOURCE_POLYGON.astype(np.int32)
    cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # 1. Run the Traffic Brain on the M4 GPU
    results = traffic_model(frame, device="mps", verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # FILTER: Only keep People(0), Cars(2), Motorcycles(3), Buses(5), Trucks(7)
    detections = detections[np.isin(detections.class_id, [0, 2, 3, 5, 7])] 
    detections = tracker.update_with_detections(detections)
    
    labels = []

    for i in range(len(detections)):
        tracker_id = detections.tracker_id[i]
        class_id = detections.class_id[i]
        x1, y1, x2, y2 = detections.xyxy[i]
        
        # Skip tracking logic for pedestrians
        if class_id == 0:
            labels.append("Person")
            continue
            
        bottom_center_x, bottom_center_y = (x1 + x2) / 2, y2
        cv2.circle(annotated_frame, (int(bottom_center_x), int(bottom_center_y)), 4, (0, 0, 255), -1)

        point = np.array([[[bottom_center_x, bottom_center_y]]], dtype=np.float32)
        real_x, real_y = cv2.perspectiveTransform(point, warp_matrix)[0][0]
        
        label_text = f"#{tracker_id}"

        if tracker_id not in vehicle_position_history:
            vehicle_position_history[tracker_id] = []
        vehicle_position_history[tracker_id].append((real_x, real_y))
        
        # Keep a 5-frame memory buffer
        if len(vehicle_position_history[tracker_id]) > 5:
            vehicle_position_history[tracker_id].pop(0)

        # --- SPEED & WRONG WAY LOGIC ---
        if len(vehicle_position_history[tracker_id]) == 5:
            old_x, old_y = vehicle_position_history[tracker_id][0]
            current_x, current_y = vehicle_position_history[tracker_id][-1]
            
            distance_meters = math.sqrt((current_x - old_x)**2 + (current_y - old_y)**2)
            speed_kmh = (distance_meters / (5 / fps)) * 3.6
            
            if speed_kmh > 5:
                # Check for Wrong Way (Y-value decreasing)
                if (current_y - old_y) < -0.5: 
                    label_text = f"#{tracker_id} WRONG WAY! ({int(speed_kmh)} km/h)"
                    
                    # AUTOMATED EVIDENCE SAVING
                    if tracker_id not in violators_caught:
                        timestamp = int(time.time())
                        filename = f"violations/WRONG_WAY_ID{tracker_id}_{timestamp}.jpg"
                        cv2.imwrite(filename, annotated_frame)
                        violators_caught[tracker_id] = True
                        print(f"[*] ENFORCEMENT ALERT: Saved {filename}")
                else:
                    label_text = f"#{tracker_id} {int(speed_kmh)} km/h"

        # --- THE CASCADE LOGIC (Motorcycles Only) ---
        if class_id == 3:
            
            # 1. TRIPLE RIDING CHECK
            rider_count = 0
            for j in range(len(detections)):
                if detections.class_id[j] == 0: 
                    px1, py1, px2, py2 = detections.xyxy[j]
                    person_center_x = (px1 + px2) / 2
                    person_center_y = (py1 + py2) / 2
                    
                    if x1 < person_center_x < x2 and y1 < person_center_y < y2:
                        rider_count += 1
            
            if rider_count > 2:
                label_text += f" [TRIPLE RIDING: {rider_count}]"
                
                # AUTOMATED EVIDENCE SAVING
                if tracker_id not in violators_caught:
                    timestamp = int(time.time())
                    filename = f"violations/TRIPLE_RIDING_ID{tracker_id}_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    violators_caught[tracker_id] = True
                    print(f"[*] ENFORCEMENT ALERT: Saved {filename}")

            # 2. HELMET DETECTION CHECK
            y1_c, y2_c = max(0, int(y1)), min(frame.shape[0], int(y2))
            x1_c, x2_c = max(0, int(x1)), min(frame.shape[1], int(x2))
            motorcycle_crop = frame[y1_c:y2_c, x1_c:x2_c]
            
            if motorcycle_crop.size > 0:
                helmet_results = helmet_model(motorcycle_crop, device="mps", verbose=False)[0]
                
                if len(helmet_results.boxes) > 0:
                    helmet_class_id = int(helmet_results.boxes[0].cls[0])
                    helmet_class_name = helmet_model.names[helmet_class_id]
                    label_text += f" [{helmet_class_name.upper()}]"

        labels.append(label_text)

    # Safely filter out the "Person" boxes so they don't clutter the screen, 
    # but keep them for the Triple Riding math above!
    display_detections = detections[detections.class_id != 0]
    display_labels = [labels[i] for i in range(len(labels)) if detections.class_id[i] != 0]

    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=display_detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=display_detections, labels=display_labels)

    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Surakshit Sadak AI - Analytics System", annotated_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()