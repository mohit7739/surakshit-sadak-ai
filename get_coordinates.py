import cv2

# Open your video
cap = cv2.VideoCapture("sample_traffic.mp4")
success, frame = cap.read()

# Resize it exactly how we resized it in the main script
frame = cv2.resize(frame, (1280, 720))

def click_event(event, x, y, flags, params):
    # If you left-click the mouse, print the X, Y coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"You clicked at: [{x}, {y}]")
        # Draw a tiny red dot where you clicked so you can see it
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Click 4 Points on the Road", frame)

cv2.imshow("Click 4 Points on the Road", frame)
cv2.setMouseCallback("Click 4 Points on the Road", click_event)

print("Click 4 points on the road to form a rectangle (Top-Left, Top-Right, Bottom-Right, Bottom-Left).")
print("Press 'q' when you are done.")

cv2.waitKey(0)
cv2.destroyAllWindows()