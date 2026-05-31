from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Detect objects
    results = model(frame)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Show detected objects
    detected_objects = []

    for result in results:
        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            confidence = float(box.conf[0])

            # Show only good confidence
            if confidence > 0.50:

                detected_objects.append(class_name)

    # Remove duplicates
    detected_objects = list(set(detected_objects))

    # Display detected objects count
    text = "Detected: " + ", ".join(detected_objects[:8])

    cv2.putText(
        annotated_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Advanced Security Surveillance",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()