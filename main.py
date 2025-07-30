# /move-mouse-with-finger/main.py

import time

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
from settings_gui import SettingsGUI

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def is_pointing(hand_landmarks):
    index_tip_y = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_mcp_y = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y
    return (middle_mcp_y - index_tip_y) > config.POINTING_THRESHOLD


def get_finger_distance(hand_landmarks, finger1, finger2):
    p1 = hand_landmarks[finger1]
    p2 = hand_landmarks[finger2]
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))


def is_fisted(hand_landmarks):
    index_tip = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
    distance = np.linalg.norm(
        np.array([index_tip.x, index_tip.y]) - np.array([wrist.x, wrist.y])
    )
    return distance < config.FIST_THRESHOLD


def main():
    # --- SETUP ---
    settings_gui = SettingsGUI()
    settings_gui.start()
    inference_engine = InferenceEngine()
    inference_engine.start()
    webcam_stream = WebcamStream().start()
    screen_width, screen_height = pyautogui.size()
    mouse = VirtualMouse(screen_width=screen_width, screen_height=screen_height)
    ui = UserInterface()

    is_mouse_on = False
    last_toggle_time = 0
    toggle_cooldown = 0.5

    current_state = State.IDLE
    prev_state = State.IDLE
    prev_time = 0
    last_known_result = None

    try:
        while True:
            # --- Frame, Time, and Inference ---
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            fps = 1 / dt if dt > 0 else 0
            frame = webcam_stream.read()
            if frame is None:
                continue
            frame = cv.flip(frame, 1)

            processing_frame = cv.resize(
                frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
            )
            rgb_processing_frame = cv.cvtColor(processing_frame, cv.COLOR_BGR2RGB)
            inference_engine.update_frame(rgb_processing_frame)
            recognition_result = inference_engine.get_latest_result()

            if recognition_result:
                last_known_result = recognition_result

            # --- Hand Data Processing ---
            left_hand, right_hand = None, None
            left_gesture = "None"
            if last_known_result and last_known_result.hand_landmarks:
                gestures = last_known_result.gestures or []
                for i, hand in enumerate(last_known_result.handedness):
                    hand_label = hand[0].category_name
                    landmarks = last_known_result.hand_landmarks[i]

                    # Get the x-coordinate of the wrist to determine its screen side
                    wrist_x = landmarks[mp.solutions.hands.HandLandmark.WRIST].x

                    # FIX: Add a spatial sanity check to prevent misidentification.
                    # This is the definitive fix for the cursor jump.

                    # A 'Left' hand must be on the left side of the screen (x < 0.5)
                    if hand_label == "Right" and wrist_x < 0.5:
                        left_hand = landmarks
                        if i < len(gestures) and gestures[i]:
                            left_gesture = gestures[i][0].category_name

                    # A 'Right' hand must be on the right side of the screen (x > 0.5)
                    elif hand_label == "Left" and wrist_x > 0.5:
                        right_hand = landmarks

            prev_state = current_state

            # --- GESTURE TOGGLE LOGIC ---
            if left_hand and left_gesture == config.NEUTRAL_GESTURE:
                if (current_time - last_toggle_time) > toggle_cooldown:
                    is_mouse_on = not is_mouse_on
                    last_toggle_time = current_time
                    print(f"Mouse control {'ACTIVATED' if is_mouse_on else 'PAUSED'}")

            # --- STATE MACHINE (only runs if mouse is toggled on) ---
            if is_mouse_on and left_hand:
                is_pinched = (
                    get_finger_distance(
                        left_hand,
                        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                        mp.solutions.hands.HandLandmark.THUMB_TIP,
                    )
                    < config.PINCH_THRESHOLD
                )
                if is_fisted(left_hand):
                    current_state = State.SCROLLING
                elif is_pinched:
                    current_state = State.DRAGGING
                elif left_gesture == config.CLICK_GESTURE:
                    pyautogui.click()
                    time.sleep(0.2)
                    current_state = State.IDLE
                else:
                    current_state = State.IDLE
            else:
                current_state = State.IDLE

            # --- State Transitions ---
            if current_state != prev_state:
                if current_state == State.DRAGGING and prev_state != State.DRAGGING:
                    pyautogui.mouseDown(button="left")
                elif prev_state == State.DRAGGING:
                    pyautogui.mouseUp(button="left")

            # --- CONTINUOUS ACTIONS ---
            if is_mouse_on:
                if right_hand:
                    right_hand_pointing = is_pointing(right_hand)
                    finger_tip = right_hand[
                        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
                    ]

                    if current_state == State.SCROLLING:
                        if mouse.is_active:
                            mouse.deactivate()
                        scroll_amount = (0.5 - finger_tip.y) * config.SCROLL_SENSITIVITY
                        pyautogui.scroll(int(scroll_amount))
                    elif right_hand_pointing and current_state in [
                        State.IDLE,
                        State.DRAGGING,
                    ]:
                        active_state = (
                            State.DRAGGING
                            if current_state == State.DRAGGING
                            else State.MOUSE_MOVEMENT
                        )
                        if not mouse.is_active:
                            mouse.activate(
                                finger_tip.x * screen_width,
                                finger_tip.y * screen_height,
                            )
                        delta = mouse.update(
                            finger_tip.x * screen_width,
                            finger_tip.y * screen_height,
                            dt,
                            active_state,
                        )
                        if delta:
                            pyautogui.move(delta[0], delta[1])
            else:
                if mouse.is_active:
                    mouse.deactivate()

            # --- Drawing ---
            final_frame = ui.draw(
                frame, last_known_result, left_gesture, current_state, fps, is_mouse_on
            )
            cv.imshow("Virtual Mouse", final_frame)

            if cv.waitKey(1) & 0xFF == 27:
                return

    finally:
        # --- CLEANUP ---
        print("Shutting down...")
        webcam_stream.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n--- AN UNEXPECTED ERROR OCCURRED ---")
        import traceback

        traceback.print_exc()
        print("--------------------------------------\n")
        input("Press Enter to exit...")
