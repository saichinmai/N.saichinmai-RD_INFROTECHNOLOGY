import cv2
import numpy as np
import time

# Load face detector model
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(
    prototxt_path,
    model_path
)

# Open webcam
cap = cv2.VideoCapture(0)

# Stability
last_label = "No Mask"
last_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    h, w = frame.shape[:2]

    # Create blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    detected_label = "No Mask"

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:

            box = detections[
                0, 0, i, 3:7
            ] * np.array([w, h, w, h])

            startX, startY, endX, endY = (
                box.astype("int")
            )

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[
                startY:endY,
                startX:endX
            ]

            if face.size == 0:
                continue

            face_height = face.shape[0]

            # Mouth + nose area
            lower_face = face[
                int(face_height * 0.45):,
                :
            ]

            gray = cv2.cvtColor(
                lower_face,
                cv2.COLOR_BGR2GRAY
            )

            # Blur
            blur = cv2.GaussianBlur(
                gray,
                (7, 7),
                0
            )

            # Texture detection
            edges = cv2.Canny(
                blur,
                30,
                100
            )

            edge_count = np.sum(edges > 0)

            # Better threshold
            # Without mask -> more facial texture
            # With mask -> smoother texture
            if edge_count < 450:
                detected_label = "Mask"
            else:
                detected_label = "No Mask"

            # Stable detection
            current_time = time.time()

            if (
                detected_label != last_label
                and current_time - last_time > 1
            ):
                last_label = detected_label
                last_time = current_time

            label = last_label

            # Colors
            if label == "Mask":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # Face box
            cv2.rectangle(
                frame,
                (startX, startY),
                (endX, endY),
                color,
                2
            )

            # Label
            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

    # Message
    if last_label == "No Mask":

        cv2.putText(
            frame,
            "ALERT: Wear Mask!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "You Wore a Mask",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    cv2.imshow(
        "Face Mask Detection System",
        frame
    )

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()