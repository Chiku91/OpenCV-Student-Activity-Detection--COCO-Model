import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Load webcam
cap = cv2.VideoCapture(0)

# Frame counter for blinking animation
frame_counter = 0

def is_standing(box, threshold=1.3):
    """Check if person is standing based on bounding box aspect ratio."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = height / width
    return aspect_ratio > threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    standing_count = 0

    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # person class
            if is_standing(box):
                standing_count += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Standing', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f'Standing Students: {standing_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show animated text if standing students detected
    if standing_count > 0 and (frame_counter // 15) % 2 == 0:
        cv2.putText(frame, "Please don't stand in class; sit down", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # Show frame
    cv2.imshow("Standing Students Detection", frame)
    frame_counter += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
