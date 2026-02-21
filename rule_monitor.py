import cv2
import math
import numpy as np
import supervision as sv
from ultralytics import YOLO

print("Loading AI Models...")
# Using the 's' model for better motorcycle accuracy
model = YOLO("yolov8s.pt") 
tracker = sv.ByteTrack()

cap = cv2.VideoCapture("sample_traffic.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# --- 1. Perspective Transformation (From your earlier calibration) ---
SOURCE_POLYGON = np.array([
    [2663, 1242], # Top-Left
    [2657, 1460], # Top-Right
    [2847, 1523], # Bottom-Right
    [2854, 1707]  # Bottom-Left
], dtype=np.float32)

TARGET_POLYGON = np.array([
    [0, 0],      
    [8, 0],      
    [8, 25],     
    [0, 25]      
], dtype=np.float32)

warp_matrix = cv2.getPerspectiveTransform(SOURCE_POLYGON, TARGET_POLYGON)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

# Dictionary to remember the real-world (X, Y) of vehicles for speed math
vehicle_positions = {}

print("Starting Advanced Bike Monitor...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Give all detections a unique Tracker ID first
    detections = tracker.update_with_detections(detections)
    
    # Filter out ONLY Motorcycles (Class 3) and Persons (Class 0)
    motorcycles = detections[detections.class_id == 3]
    persons = detections[detections.class_id == 0]

    labels = []
    
    # Analyze each motorcycle individually
    for i in range(len(motorcycles)):
        tracker_id = motorcycles.tracker_id[i]
        mx1, my1, mx2, my2 = motorcycles.xyxy[i]
        
        # --- FEATURE A: Triple Riding Logic ---
        riders_on_this_bike = 0
        for j in range(len(persons)):
            px1, py1, px2, py2 = persons.xyxy[j]
            person_center_x = (px1 + px2) / 2
            person_center_y = (py1 + py2) / 2
            
            # Is the person sitting on THIS motorcycle?
            if (mx1 < person_center_x < mx2) and (my1 < person_center_y < my2):
                riders_on_this_bike += 1
                
        # --- FEATURE B: Speed Logic ---
        # Track the bottom center of the motorcycle tire
        bottom_center_x = (mx1 + mx2) / 2
        bottom_center_y = my2
        
        # Convert screen pixels to Real-World Meters
        point = np.array([[[bottom_center_x, bottom_center_y]]], dtype=np.float32)
        real_world_point = cv2.perspectiveTransform(point, warp_matrix)
        real_x, real_y = real_world_point[0][0]
        
        speed_kmh = 0
        if tracker_id in vehicle_positions:
            prev_x, prev_y = vehicle_positions[tracker_id]
            # Calculate distance in meters, then convert to km/h
            distance_meters = math.sqrt((real_x - prev_x)**2 + (real_y - prev_y)**2)
            speed_kmh = (distance_meters / (1 / fps)) * 3.6
            
        # Save current position for the next frame
        vehicle_positions[tracker_id] = (real_x, real_y)
        
        # --- FEATURE C: Combine the Text Label ---
        label_text = f"#{tracker_id} "
        
        # Add Speed to the label (only if it's moving)
        if speed_kmh > 5:
            label_text += f" {int(speed_kmh)}km/h | "
            
        # Add Triple Riding Alert to the label
        if riders_on_this_bike > 2:
            label_text += f"TRIPLE RIDING!"
        else:
            label_text += f"Riders: {riders_on_this_bike}"
            
        labels.append(label_text)

    # Draw everything on the screen
    annotated_frame = frame.copy()
    
    # Draw the yellow speed zone box
    pts = SOURCE_POLYGON.astype(np.int32)
    cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    
    # Draw boxes and labels only on the motorcycles
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=motorcycles)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=motorcycles, labels=labels)

    cv2.imshow("Surakshit Sadak AI - Bike Speed & Rules", annotated_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()