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

pyautogui.FAILSAFE = False


def main():
    # --- SETUP ---
    vision = Vision()
    webcam_stream = WebcamStream().start()
    screen_width, screen_height = pyautogui.size()
    mouse = VirtualMouse(screen_width=screen_width, screen_height=screen_height)
    ui = UserInterface()
    # FPS Counter
    prev_time = 0
    current_time = 0

    # Cooldown for click action to prevent spamming
    click_cooldown = 0.5  # seconds
    last_click_time = 0

    needs_reanchor = True
    # --- MAIN LOOP ---
    while True:
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        frame = webcam_stream.read()
        if frame is None:
            continue

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        recognition_result = vision.recognize(rgb_frame)

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
                        finger_tip.x * screen_width, finger_tip.y * screen_height
                    )
                    if delta:
                        pyautogui.move(delta[0], delta[1])

            elif gesture_name in config.ACTION_GESTURES:
                # Any action gesture deactivates mouse movement.
                if mouse.is_active:
                    mouse.deactivate()

                # Handle specific actions like clicking
                if gesture_name == "Thumb_Up":
                    current_time = time.time()
                    if current_time - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = current_time
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

    # --- CLEANUP ---
    webcam_stream.release()  # Use the release method you created
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
