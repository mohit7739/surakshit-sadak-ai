import cv2

# 1. Load the video and grab just the very first frame
cap = cv2.VideoCapture("sample_traffic.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read the video file. Check the filename.")
    exit()

# 2. Define what happens when you click your mouse
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the exact format we need for our tracker.py file
        print(f"    [{x}, {y}],")
        
        # Draw a small red dot where you clicked so you can see your zone
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Surakshit Sadak AI - Calibrator", frame)

# 3. Open the window and listen for mouse clicks
print("📸 CALIBRATION TOOL STARTED")
print("Click 4 points on the road in this exact order to form your zone:")
print("1. Top-Left")
print("2. Top-Right")
print("3. Bottom-Right")
print("4. Bottom-Left")
print("Press the 'q' key on your keyboard when you are finished.")

cv2.imshow("Surakshit Sadak AI - Calibrator", frame)
cv2.setMouseCallback("Surakshit Sadak AI - Calibrator", get_coordinates)

# Keep the window open until 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()