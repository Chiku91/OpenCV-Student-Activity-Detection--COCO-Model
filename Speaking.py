import cv2
import mediapipe as mp
import torch
import numpy as np
import sounddevice as sd
import queue

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10)

# Video capture
cap = cv2.VideoCapture(0)

# Thresholds
VOLUME_THRESHOLD = 0.0005  # Tune this as needed

# Audio processing variables
audio_queue = queue.Queue()
current_volume = 0.0

def audio_callback(indata, frames, time_info, status):
    """Audio stream callback: calculates volume and puts in queue."""
    volume_norm = np.linalg.norm(indata) / frames
    audio_queue.put(volume_norm)

# Start audio input stream (non-blocking)
stream = sd.InputStream(channels=1, samplerate=44100, callback=audio_callback)
stream.start()

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Update current_volume from queue if available
    while not audio_queue.empty():
        current_volume = audio_queue.get()

    audio_detected = current_volume > VOLUME_THRESHOLD

    # Detect people using YOLOv5
    results = model(rgb_frame)
    persons = []
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # person class
            x1, y1, x2, y2 = map(int, box)
            persons.append((x1, y1, x2, y2))

    # Face mesh detection
    face_results = face_mesh.process(rgb_frame)
    speaking_students = 0

    if face_results.multi_face_landmarks and audio_detected:
        for face_landmarks in face_results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            cx, cy = int(nose.x * w), int(nose.y * h)
            speaking_students += 1
            cv2.putText(frame, "Student is speaking", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    # Draw rectangles around detected persons
    for x1, y1, x2, y2 in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display number of speaking students
    cv2.putText(frame, f'Speaking Students: {speaking_students}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display current audio volume for debug
    cv2.putText(frame, f'Audio Volume: {current_volume:.4f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Student Speaking Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()
