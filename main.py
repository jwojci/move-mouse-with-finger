import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

import pyautogui
import cv2 as cv

import config
from mouse import VirtualMouse
from camera import WebcamStream
from vision import Vision
from ui import UserInterface
from engine import InferenceEngine

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def main():
    # --- SETUP ---
    inference_engine = InferenceEngine()
    inference_engine.start()

    vision = Vision()
    webcam_stream = WebcamStream().start()
    screen_width, screen_height = pyautogui.size()
    mouse = VirtualMouse(screen_width=screen_width, screen_height=screen_height)
    ui = UserInterface()

    # FPS Counter
    prev_time = 0

    # Cooldown for click action to prevent spamming
    click_cooldown = 0.5  # seconds
    last_click_time = 0

    last_known_result = None

    # --- MAIN LOOP ---
    try:
        while True:
            current_time = time.time()
            dt = current_time - prev_time  # <-- Calculate delta time (dt)
            prev_time = current_time

            if dt > 0:
                fps = 1 / dt
            else:
                fps = 0

            frame = webcam_stream.read()
            if frame is None:
                continue

            frame = cv.flip(frame, 1)
            # 1. Create a smaller frame for AI processing
            processing_frame = cv.resize(
                frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
            )

            # 2. Convert the SMALLER frame to RGB
            rgb_processing_frame = cv.cvtColor(processing_frame, cv.COLOR_BGR2RGB)

            # 3. Give the SMALLER frame to the model
            recognition_result = vision.recognize(rgb_processing_frame)

            gesture_name = "None"
            if recognition_result and recognition_result.gestures:
                top_gesture = recognition_result.gestures[0][0]
                gesture_name = top_gesture.category_name if top_gesture else "None"

                # If a hand and gesture are detected, process them
                if gesture_name in config.NEUTRAL_GESTURES:
                    finger_tip = recognition_result.hand_landmarks[0][
                        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
                    ]
                    if not mouse.is_active:
                        mouse.activate(
                            finger_tip.x * screen_width, finger_tip.y * screen_height
                        )
                    else:
                        delta = mouse.update(
                            finger_tip.x * screen_width,
                            finger_tip.y * screen_height,
                            dt,
                        )
                        if delta:
                            pyautogui.move(delta[0], delta[1])
                elif gesture_name in config.ACTION_GESTURES:
                    # Any action gesture deactivates mouse movement.
                    if mouse.is_active:
                        mouse.deactivate()
                    if gesture_name == "Thumb_Up":
                        if current_time - last_click_time > click_cooldown:
                            pyautogui.click()
                            last_click_time = time.time()
            else:
                # No hand detected, deactivate mouse.
                if mouse.is_active:
                    mouse.deactivate()

            # --- DRAWING ---
            # Use the UI class to draw all feedback on the frame
            final_frame = ui.draw(
                frame, recognition_result, gesture_name, mouse.is_active, fps
            )

            cv.imshow("Virtual Mouse", final_frame)
            if cv.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break
    finally:
        # --- CLEANUP ---
        inference_engine.stop()
        webcam_stream.release()  # Use the release method you created
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
