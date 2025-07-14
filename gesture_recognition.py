import time

import mediapipe as mp
import cv2 as cv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


base_options = python.BaseOptions(model_asset_path="model\gesture_recognizer.task")
options = vision.GestureRecognizerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=1
)
recognizer = vision.GestureRecognizer.create_from_options(options)


cap = cv.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    # Convert the BGR image to RGB and create a MediaPipe Image.
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Get a timestamp for the frame and recognize gestures.
    timestamp_ms = int(time.time() * 1000)
    recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    # If a gesture is recognized, draw it on the frame.
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        gesture_name = top_gesture.category_name

        # Use OpenCV to draw the gesture name on the frame.
        cv.putText(
            frame,
            f"Gesture: {gesture_name}",
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

    # Display the frame.
    cv.imshow("Gesture Recognition", frame)

    # Break the loop if 'q' is pressed.
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# --- CLEANUP ---
cap.release()
cv.destroyAllWindows()
