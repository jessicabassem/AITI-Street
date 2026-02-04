from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("codes/yolo11s.pt")  # change path if needed

# Open laptop camera (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO inference
    results = model(frame, conf=0.4)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Webcam Inference", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
