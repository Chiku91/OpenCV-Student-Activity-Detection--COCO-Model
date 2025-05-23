import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load YOLOv8 model
yolo = YOLO("yolov8n.pt")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5)
mp_hdraw = mp.solutions.drawing_utils

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)
mp_pdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Create animation frames
warning_text = "Sit up straight and don't keep your head down."
animation_frames = []
for i in range(15):
    anim_frame = np.zeros((80, 700, 3), dtype=np.uint8)
    color = (0, 0, 255 - i * 15) if i < 8 else (0, 0, (i - 7) * 15)
    cv2.putText(anim_frame, warning_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    animation_frames.append(anim_frame)

animation_index = 0
show_warning_animation = False

def hand_box_from_landmarks(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    results = list(yolo.track(source=frame, persist=True, stream=False, verbose=False))
    person_boxes = []
    show_warning_animation = False  # reset flag per frame

    for r in results:
        if not (r.boxes and r.boxes.xyxy is not None and r.boxes.cls is not None):
            continue
        ids_tensor = r.boxes.id
        ids = ids_tensor.tolist() if ids_tensor is not None else [None] * len(r.boxes.xyxy)
        for box, cls, tid in zip(r.boxes.xyxy, r.boxes.cls, ids):
            if yolo.names[int(cls)] == "person" and tid is not None:
                x1, y1, x2, y2 = map(int, box)
                person_boxes.append((int(tid), x1, y1, x2, y2))

    for pid, x1, y1, x2, y2 in person_boxes:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        h, w = roi.shape[:2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Pose analysis
        pose_res = pose.process(roi_rgb)
        head_down = False
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            nose_y = lm[mp_pose.PoseLandmark.NOSE].y * h
            ls_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
            rs_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
            shoulder_mid_y = (ls_y + rs_y) / 2
            if nose_y > shoulder_mid_y + 0.05 * h:
                head_down = True
            mp_pdraw.draw_landmarks(roi, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Hand analysis
        eye_level = int(0.25 * h)
        cv2.line(frame, (x1, y1 + eye_level), (x2, y1 + eye_level), (200, 200, 200), 2)

        hands_res = hands.process(roi_rgb)
        writing = False
        if hands_res.multi_hand_landmarks:
            for hl in hands_res.multi_hand_landmarks:
                mp_hdraw.draw_landmarks(roi, hl, mp_hands.HAND_CONNECTIONS)
                hx1, hy1, hx2, hy2 = hand_box_from_landmarks(hl.landmark, w, h)
                if hy2 > eye_level:
                    writing = True
                    break

        if writing and head_down:
            label = f"ID {pid}: Writing (head down)"
            color = (0, 255, 0)
            show_warning_animation = True
        else:
            label = f"ID {pid}: Not Writing"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Show animation if triggered
    if show_warning_animation:
        anim_frame = animation_frames[animation_index]
        animation_index = (animation_index + 1) % len(animation_frames)
        anim_resized = cv2.resize(anim_frame, (min(frame_w - 20, 700), 80))
        y_offset = frame_h - 90
        x_offset = (frame_w - anim_resized.shape[1]) // 2
        frame[y_offset:y_offset + anim_resized.shape[0], x_offset:x_offset + anim_resized.shape[1]] = anim_resized

    cv2.imshow("Writing with Head Down Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
