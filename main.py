import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pyautogui
import cv2 as cv

import config
from mouse import VirtualMouse
from camera import WebcamStream

pyautogui.FAILSAFE = False


def draw_ui(frame, hand_results, margin):
    # Draw the bounding box
    frame_height, frame_width, _ = frame.shape
    start_x = int(margin * frame_width)
    start_y = int(margin * frame_height)
    end_x = frame_width - start_x
    end_y = frame_height - start_y
    cv.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    # Draw the hand landmarks
    if hand_results.multi_hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_hand_styles = mp.solutions.drawing_styles
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_hand_styles.get_default_hand_landmarks_style(),
                mp_hand_styles.get_default_hand_connections_style(),
            )


def main():
    # --- SETUP ---
    # 1. Initialize Gesture Recognizer
    base_options = python.BaseOptions(model_asset_path="model/gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # 2. Initialize other components
    webcam_stream = WebcamStream().start()
    screen_width, screen_height = pyautogui.size()
    mouse = VirtualMouse(screen_width=screen_width, screen_height=screen_height)

    # 3. Cooldown for click action to prevent spamming
    click_cooldown = 0.5  # seconds
    last_click_time = 0

    needs_reanchor = True
    # --- MAIN LOOP ---
    while True:
        frame = webcam_stream.read()
        if frame is None:
            continue

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Recognize gesture
        timestamp_ms = int(time.time() * 1000)
        recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            gesture_name = top_gesture.category_name if top_gesture else "None"
            hand_landmarks = recognition_result.hand_landmarks[0]
            finger_tip = hand_landmarks[
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
            ]

            # If a hand and gesture are detected, process them
            if gesture_name in config.ACTION_GESTURES:
                needs_reanchor = True

                if gesture_name == "Thumb_Up":
                    current_time = time.time()
                    if current_time - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = current_time

            elif gesture_name in config.NEUTRAL_GESTURES:
                if needs_reanchor:
                    mouse.activate(
                        finger_tip.x * screen_width, finger_tip.y * screen_height
                    )
                    needs_reanchor = False
                else:
                    delta = mouse.update(
                        finger_tip.x * screen_width, finger_tip.y * screen_height
                    )
                    if delta:
                        pyautogui.move(delta[0], delta[1])
            cv.putText(
                frame,
                gesture_name,
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

        else:
            # 4. If NO hand is detected, make sure we re-anchor next time
            needs_reanchor = True

        cv.imshow("Virtual Mouse", frame)
        if cv.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    # --- CLEANUP ---
    webcam_stream.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
