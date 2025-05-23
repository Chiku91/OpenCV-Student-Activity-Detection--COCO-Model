# Student Activity Detection System 🎓📚

A smart classroom monitoring system using OpenCV and pretrained COCO model to detect various student activities in real-time. It enhances classroom engagement by dynamically providing visual feedback and alerts based on students' behavior.

## Features ✨

1. **Face Detection - Student Attending**  
   Detects when a student’s face is visible, indicating attentiveness. When detected, dynamic visual content like graphs and pie charts are displayed to aid study engagement.  
   ![Face Detection GIF](path_to_face_detection.gif) 👩‍🎓📊

2. **Hand Raise Detection - Ask Your Doubts**  
   Detects when a student raises their hand and displays an animated message: "You can ask your doubts". Encourages active participation.  
   ![Hand Raise GIF](path_to_hand_raise.gif) ✋❓

3. **Writing Detection & Device Usage Warning**  
   Detects if a student is writing. If phone or laptop usage is detected, an alert "Don't use phone/laptop" is displayed to minimize distractions.  
   ![Writing Detection GIF](path_to_writing_detection.gif) 📝📵

4. **Posture Alert - Head Down Warning**  
   Detects if a student is writing but keeping their head down, prompting the message: "Don't keep your head down, sit straight up" to encourage good posture.  
   ![Posture Alert GIF](path_to_posture_alert.gif) 🙆‍♂️🪑

5. **Listening vs Talking Detection**  
   Differentiates if a student is listening or talking during the lesson. Animated emojis are displayed to help regain student focus.  
   ![Listening vs Talking GIF](path_to_listening_talking.gif) 👂🗣️😊

6. **Lapse in Concentration Detection**  
   Detects if a student is inattentive by turning their head away and displays appropriate alert messages to help them refocus.  
   ![Concentration Alert GIF](path_to_concentration_alert.gif) 🚨🧠

7. **Teacher Guidance Detection**  
   Detects interactions between teacher and student. Displays encouraging message "You can ask your doubts" when a teacher is guiding a student.  
   ![Teacher Guidance GIF](path_to_teacher_guidance.gif) 👩‍🏫🤝

## How It Works ⚙️

- Uses OpenCV’s pretrained COCO model for object detection and classification.  
- Processes real-time video feed from classroom cameras.  
- Applies custom logic on detected classes like faces, hands, laptops, phones, and writing postures.  
- Displays dynamic visual feedback and animated messages to improve student engagement and classroom discipline.

## Installation 🚀

1. Clone the repository:  
   ```bash
   git clone https://github.com/Chiku-91/student-activity-detection.git
