import time
from enum import Enum

import pyautogui
import cv2 as cv
import numpy as np
import mediapipe as mp

import config
from utils import State
from camera import WebcamStream
from engine import InferenceEngine
from mouse import VirtualMouse
from ui import UserInterface

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def get_finger_distance(hand_landmarks, finger1, finger2):
    p1 = hand_landmarks[finger1]
    p2 = hand_landmarks[finger2]
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))


def main():
    # --- SETUP ---
    inference_engine = InferenceEngine()
    inference_engine.start()

    webcam_stream = WebcamStream().start()
    screen_width, screen_height = pyautogui.size()
    mouse = VirtualMouse(screen_width=screen_width, screen_height=screen_height)
    ui = UserInterface()
    current_state = State.IDLE

    # FPS Counter
    prev_time = 0
    last_known_result = None

    pinch_distance = 0
    pinch_enter_counter = 0
    pinch_exit_counter = 0

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

            # --- Inference ---
            processing_frame = cv.resize(
                frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
            )
            rgb_processing_frame = cv.cvtColor(processing_frame, cv.COLOR_BGR2RGB)
            inference_engine.update_frame(rgb_processing_frame)
            recognition_result = inference_engine.get_latest_result()

            if recognition_result:
                last_known_result = recognition_result

            # --- State Machine Logic ---
            gesture_name = "None"
            did_state_change = False
            is_hand_present = (
                last_known_result
                and last_known_result.gestures
                and last_known_result.gestures[0]
            )

            if is_hand_present:
                gesture_name = last_known_result.gestures[0][0].category_name
                hand_landmarks = recognition_result.hand_landmarks[0]

                pinch_distance = get_finger_distance(
                    hand_landmarks,
                    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                    mp.solutions.hands.HandLandmark.THUMB_TIP,
                )
                is_pinched = pinch_distance < config.PINCH_THRESHOLD
                is_movement_gesture = gesture_name in config.MOVEMENT_GESTURES

                # --- Hysteresis Counter ---
                if is_pinched:
                    pinch_enter_counter += 1
                    pinch_exit_counter = 0
                else:
                    pinch_enter_counter = 0
                    pinch_exit_counter += 1

                if current_state == State.IDLE:
                    if (
                        is_movement_gesture
                        and pinch_exit_counter > config.GESTURE_CONFIRMATION_FRAMES
                    ):
                        current_state = State.MOUSE_MOVEMENT
                        did_state_change = True

                elif current_state == State.MOUSE_MOVEMENT:
                    if pinch_enter_counter > config.GESTURE_CONFIRMATION_FRAMES:
                        current_state = State.DRAGGING
                        pyautogui.mouseDown(button="left")
                        did_state_change = True
                        print("State Change -> DRAGGING")
                    elif not is_movement_gesture:
                        current_state = State.IDLE
                        mouse.deactivate()
                        did_state_change = True

                elif current_state == State.DRAGGING:
                    if pinch_exit_counter > config.GESTURE_CONFIRMATION_FRAMES:
                        current_state = State.IDLE
                        pyautogui.mouseUp(button="left")
                        mouse.deactivate()
                        did_state_change = True
                        print("State Change -> IDLE (from Drag)")

                # --- Execute State Behavior ---
                # ONLY process movement if the state did NOT change in this frame.
                # This prevents the cursor from jumping when a drag starts.
                if not did_state_change and current_state in [
                    State.MOUSE_MOVEMENT,
                    State.DRAGGING,
                ]:
                    finger_tip = hand_landmarks[
                        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
                    ]
                    if not mouse.is_active:
                        mouse.activate(
                            finger_tip.x * screen_width, finger_tip.y * screen_height
                        )

                    delta = mouse.update(
                        finger_tip.x * screen_width,
                        finger_tip.y * screen_height,
                        dt,
                        current_state,
                    )
                    if delta:
                        pyautogui.move(delta[0], delta[1])

            else:  # No hand detected
                if mouse.is_active or current_state == State.DRAGGING:
                    pyautogui.mouseUp(button="left")
                    mouse.deactivate()
                current_state = State.IDLE

            # --- DRAWING ---
            final_frame = ui.draw(
                frame,
                last_known_result,
                gesture_name,
                current_state,
                fps,
            )
            cv.putText(
                final_frame,
                f"Pinch: {pinch_distance:.3f}",
                (10, 170),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
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
