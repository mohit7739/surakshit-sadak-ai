import cv2
from ultralytics import YOLO

# Load your new 100-epoch production model!
model = YOLO('surakshit_100_epochs.pt')

# Open the MacBook's default webcam
cap = cv2.VideoCapture(0)

print("Starting Surakshit Sadak AI Webcam... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # NEW TRACKING LINE: Tells YOLO to assign a persistent ID to every helmet
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("Surakshit Sadak AI - Live Test", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()