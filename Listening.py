import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Capture from webcam
cap = cv2.VideoCapture(0)

# Track attention
attention_start_time = None
attention_duration = 0
last_detected = False
ATTENTION_THRESHOLD = 20  # seconds
MOVEMENT_THRESHOLD = 5000

prev_gray = None

# Create animated warning frames
warning_text = "Concentrate! Listen carefully and ask doubts!"
animation_frames = []
for i in range(15):
    frame = np.zeros((80, 800, 3), dtype=np.uint8)
    color = (0, 0, 255 - i * 15) if i < 8 else (0, 0, (i - 7) * 15)
    cv2.putText(frame, warning_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    animation_frames.append(frame)

animation_index = 0
show_warning = False

# --- Helper functions for feedback visuals ---

def create_emoji_feedback():
    feedback = {
        "Attentive": "🙂",
        "Bored": "😑",
        "Distracted": "😵"
    }
    # Create one combined feedback image with all emojis vertically stacked
    img = np.zeros((250 * len(feedback), 400, 3), dtype=np.uint8)
    y_offset = 0
    for label, emoji in feedback.items():
        cv2.putText(img, emoji, (160, y_offset + 140), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 0), 2)
        cv2.putText(img, label, (140, y_offset + 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 250
    return img

def create_progress_bar(progress):
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    bar_width = int((progress / 100) * 300)
    cv2.rectangle(img, (50, 50), (350, 80), (100, 100, 100), -1)
    cv2.rectangle(img, (50, 50), (50 + bar_width, 80), (0, 128, 255), -1)
    cv2.putText(img, f"{progress}%", (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

def create_blinking_alert(frame_index):
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    color = (0, 0, 255) if frame_index % 2 == 0 else (0, 0, 0)
    cv2.putText(img, "NOT ATTENTIVE!", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
    return img

def draw_meter(attention_level):
    bar_width = int((attention_level / 100) * 300)
    img = np.zeros((150, 400, 3), dtype=np.uint8)
    if attention_level >= 70:
        color = (0, 255, 0)
    elif attention_level >= 40:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    cv2.rectangle(img, (50, 60), (350, 100), (50, 50, 50), -1)
    cv2.rectangle(img, (50, 60), (50 + bar_width, 100), color, -1)
    cv2.putText(img, f"Attention: {attention_level}%", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img

animation_feedback_index = 0
progress = 0
progress_direction = 1
blink_frame = 0
attention_meter_level = 0
attention_meter_direction = 1

feedback_window_open = False  # Flag to track if feedback window is open

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    current_time = time.time()
    status = "Not attentive"
    movement_detected = False

    # Movement detection
    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        movement_amount = np.sum(thresh)
        if movement_amount > MOVEMENT_THRESHOLD:
            movement_detected = True
    prev_gray = gray.copy()

    show_warning = False  # reset flag each frame

    if result.multi_face_landmarks and not movement_detected:
        if not last_detected:
            attention_start_time = current_time
            attention_duration = 0
        else:
            attention_duration = current_time - attention_start_time

        if attention_duration >= ATTENTION_THRESHOLD:
            status = "Attentive"
        else:
            status = "Listening to lesson"
        last_detected = True
    else:
        attention_start_time = None
        attention_duration = 0
        last_detected = False
        show_warning = True
        if movement_detected:
            status = "Not concentrating"
        else:
            status = "Not attentive"

    # Display status on webcam frame
    color = (0, 255, 0) if status == "Attentive" else (0, 0, 255)
    cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show animated warning if not attentive
    if show_warning:
        anim_frame = animation_frames[animation_index]
        animation_index = (animation_index + 1) % len(animation_frames)
        anim_resized = cv2.resize(anim_frame, (min(w - 20, 700), 80))
        y_offset = h - 90
        x_offset = (w - anim_resized.shape[1]) // 2
        frame[y_offset:y_offset + anim_resized.shape[0], x_offset:x_offset + anim_resized.shape[1]] = anim_resized

    # Show the main webcam feed
    cv2.imshow("Attention Detector", frame)

    # Show feedback visuals only when NOT concentrating or NOT attentive
    if status in ["Not concentrating", "Not attentive"]:
        # Emoji feedback (static combined image)
        emoji_img = create_emoji_feedback()

        # Progress bar logic - progress goes back and forth
        progress += progress_direction * 2
        if progress >= 100:
            progress = 100
            progress_direction = -1
        elif progress <= 0:
            progress = 0
            progress_direction = 1
        progress_img = create_progress_bar(progress)

        # Blinking alert toggling
        blink_img = create_blinking_alert(blink_frame)
        blink_frame = (blink_frame + 1) % 20

        # Attention meter logic - goes up and down
        attention_meter_level += attention_meter_direction * 5
        if attention_meter_level >= 100:
            attention_meter_level = 100
            attention_meter_direction = -1
        elif attention_meter_level <= 0:
            attention_meter_level = 0
            attention_meter_direction = 1
        meter_img = draw_meter(attention_meter_level)

        # Combine all feedback visuals vertically
        combined_height = emoji_img.shape[0] + progress_img.shape[0] + blink_img.shape[0] + meter_img.shape[0]
        combined_width = max(emoji_img.shape[1], progress_img.shape[1], blink_img.shape[1], meter_img.shape[1])
        combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        y = 0
        for img_part in [emoji_img, progress_img, blink_img, meter_img]:
            combined_img[y:y+img_part.shape[0], 0:img_part.shape[1]] = img_part
            y += img_part.shape[0]

        cv2.imshow("Attention Feedback", combined_img)
        feedback_window_open = True  # Mark feedback window as open

    else:
        # Close feedback window if open when attentive or listening
        if feedback_window_open:
            cv2.destroyWindow("Attention Feedback")
            feedback_window_open = False  # Mark feedback window as closed

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
