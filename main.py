from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Object tracking
    results = model.track(frame, persist=True)

    # Draw detections
    annotated_frame = results[0].plot()

    # Count persons
    person_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])

        # Class 0 = person in COCO dataset
        if cls == 0:
            person_count += 1

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display person count
    cv2.putText(
        annotated_frame,
        f"Persons: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Show output
    cv2.imshow("AI Object Detection & Tracking", annotated_frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()