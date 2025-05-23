import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Initialize YOLOv8 (person detector)
yolo_model = YOLO("yolov8n.pt")  # Ensure you have YOLOv8 installed

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Create animation frames for message (simulate GIF)
animation_frames = []
for i in range(15):
    frame = np.zeros((80, 450, 3), dtype=np.uint8)
    # Color changes for simple animation effect
    color = (0, 255 - i*15, i*15)
    cv2.putText(frame, "You may ask your doubts", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    animation_frames.append(frame)

animation_index = 0
show_animation = False
animation_counter = 0  # counts frames to show animation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO to detect persons
    results = yolo_model(frame)
    person_boxes = []
    hand_raised_detected = False

    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if yolo_model.names[int(cls)] == "person":
                x1, y1, x2, y2 = map(int, box)
                person_boxes.append((x1, y1, x2, y2))

    # For each detected person
    for (x1, y1, x2, y2) in person_boxes:
        person_roi = frame[y1:y2, x1:x2]

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # If hands detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(person_roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist_y = hand_landmarks.landmark[0].y
                middle_tip_y = hand_landmarks.landmark[12].y

                if middle_tip_y < wrist_y:
                    hand_raised_detected = True
                    cv2.putText(frame, "Hand Raised", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show animation if hand raised recently
    if hand_raised_detected:
        show_animation = True
        animation_counter = 30  # show animation for ~30 frames (~1 sec at 30 FPS)

    if show_animation:
        anim_frame = animation_frames[animation_index]
        animation_index = (animation_index + 1) % len(animation_frames)

        # Resize animation frame (optional)
        anim_resized = cv2.resize(anim_frame, (450, 80))

        # Overlay animation on bottom of frame
        h, w, _ = frame.shape
        y_offset = h - 90
        x_offset = 10

        # Overlay the animation (simple direct copy; assumes black bg)
        frame[y_offset:y_offset + anim_resized.shape[0], x_offset:x_offset + anim_resized.shape[1]] = anim_resized

        animation_counter -= 1
        if animation_counter <= 0:
            show_animation = False

    cv2.imshow("Multiple Hand Raise Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
