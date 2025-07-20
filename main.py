import time

import pyautogui
import cv2 as cv
import mediapipe as mp

import config
from camera import WebcamStream
from engine import InferenceEngine
from mouse import VirtualMouse
from ui import UserInterface

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def main():
    # --- SETUP ---
    # The InferenceEngine now manages the Vision model
    inference_engine = InferenceEngine()
    inference_engine.start()

    webcam_stream = WebcamStream().start()
    screen_width, screen_height = pyautogui.size()
    mouse = VirtualMouse(screen_width=screen_width, screen_height=screen_height)
    ui = UserInterface()

    # FPS Counter
    prev_time = 0

    # Cooldown for click action to prevent spamming
    click_cooldown = 0.5  # seconds
    last_click_time = 0

    # --- MAIN LOOP ---
    try:
        while True:
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            fps = 1 / dt if dt > 0 else 0

            # 1. Get the latest frame from the camera thread
            frame = webcam_stream.read()
            if frame is None:
                continue

            frame = cv.flip(frame, 1)

            # 2. Prepare frame and send it to the inference engine (Producer)
            processing_frame = cv.resize(
                frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
            )
            rgb_processing_frame = cv.cvtColor(processing_frame, cv.COLOR_BGR2RGB)
            inference_engine.update_frame(rgb_processing_frame)

            # 3. Get the latest result from the inference engine (Consumer)
            # This is non-blocking and returns the most recent result available.
            recognition_result = inference_engine.get_latest_result()

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
                            finger_tip.x * config.WEBCAM_WIDTH,
                            finger_tip.y * config.WEBCAM_HEIGHT,
                        )
                    else:
                        # Use the original webcam resolution for coordinate mapping
                        target_x = finger_tip.x * config.WEBCAM_WIDTH
                        target_y = finger_tip.y * config.WEBCAM_HEIGHT

                        delta = mouse.update(target_x, target_y, dt)
                        if delta:
                            pyautogui.move(delta[0], delta[1])

                elif gesture_name in config.ACTION_GESTURES:
                    # Any action gesture deactivates mouse movement.
                    mouse.deactivate()
                    if gesture_name == "Thumb_Up":
                        if current_time - last_click_time > click_cooldown:
                            pyautogui.click()
                            last_click_time = current_time
            else:
                # No hand detected, deactivate mouse.
                mouse.deactivate()

            # --- DRAWING ---
            final_frame = ui.draw(
                frame, recognition_result, gesture_name, mouse.is_active, fps
            )

            cv.imshow("Virtual Mouse", final_frame)
            if cv.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break
    finally:
        # --- CLEANUP ---
        print("Shutting down...")
        inference_engine.stop()
        webcam_stream.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
