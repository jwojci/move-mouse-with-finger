import time
import traceback

import pyautogui
import cv2 as cv
import mediapipe as mp
from loguru import logger

import config
import gestures
from utils import State
from camera import WebcamStream
from engine import InferenceEngine
from mouse import VirtualMouse
from ui import UserInterface
from settings_gui import SettingsGUI

# --- Configurations ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
logger.add("main.log", rotation="10 MB")


class MouseCamApp:
    """
    The main application class that orchestrates the camera, inference engine,
    mouse control, and user interface.
    """

    def __init__(self):
        """Initializes all components of the application."""
        logger.info("Initializing MouseCamApp...")
        self.settings_gui = SettingsGUI()
        self.settings_gui.start()

        self.inference_engine = InferenceEngine()
        self.inference_engine.start()

        self.webcam_stream = WebcamStream().start()
        self.screen_width, self.screen_height = pyautogui.size()
        self.mouse = VirtualMouse(
            screen_width=self.screen_width, screen_height=self.screen_height
        )
        self.ui = UserInterface()

        # --- Application State ---
        self.is_mouse_on = False
        self.last_toggle_time = 0
        self.toggle_cooldown = 0.5  # Seconds

        self.current_state = State.IDLE
        self.prev_state = State.IDLE

        self.last_known_result = None
        self.prev_time = time.time()

    def run(self):
        """The main application loop."""
        try:
            while True:
                # 1. --- Frame, Time, and Inference ---
                current_time = time.time()
                dt = current_time - self.prev_time
                self.prev_time = current_time
                fps = 1 / dt if dt > 0 else 0

                frame = self.webcam_stream.read()
                if frame is None:
                    continue
                frame = cv.flip(frame, 1)

                # Prepare frame for inference
                processing_frame = cv.resize(
                    frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
                )
                rgb_processing_frame = cv.cvtColor(processing_frame, cv.COLOR_BGR2RGB)
                self.inference_engine.update_frame(rgb_processing_frame)
                recognition_result = self.inference_engine.get_latest_result()

                if recognition_result:
                    self.last_known_result = recognition_result

                # 2. --- Hand Data Processing ---
                left_hand, right_hand = None, None
                left_gesture_str = "None"
                if self.last_known_result and self.last_known_result.hand_landmarks:
                    detected_gestures = self.last_known_result.gestures or []
                    for i, hand in enumerate(self.last_known_result.handedness):
                        hand_label = hand[0].category_name
                        landmarks = self.last_known_result.hand_landmarks[i]
                        wrist_x = landmarks[mp.solutions.hands.HandLandmark.WRIST].x

                        # Spatial check to prevent hand misidentification
                        if hand_label == "Right" and wrist_x < 0.5:
                            left_hand = landmarks
                            if i < len(detected_gestures) and detected_gestures[i]:
                                left_gesture_str = detected_gestures[i][0].category_name
                        elif hand_label == "Left" and wrist_x > 0.5:
                            right_hand = landmarks

                self.prev_state = self.current_state

                # 3. --- GESTURE TOGGLE LOGIC ---
                if left_hand and left_gesture_str == config.NEUTRAL_GESTURE:
                    if (current_time - self.last_toggle_time) > self.toggle_cooldown:
                        self.is_mouse_on = not self.is_mouse_on
                        self.last_toggle_time = current_time
                        logger.info(
                            f"Mouse control {'ACTIVATED' if self.is_mouse_on else 'PAUSED'}"
                        )

                # 4. --- STATE MACHINE ---
                if self.is_mouse_on and left_hand:
                    if gestures.is_fisted(left_hand):
                        self.current_state = State.SCROLLING
                    elif gestures.is_pinched(left_hand):
                        self.current_state = State.DRAGGING
                    elif left_gesture_str == config.CLICK_GESTURE:
                        pyautogui.click()
                        time.sleep(0.2)
                        self.current_state = State.IDLE
                    else:
                        self.current_state = State.IDLE
                else:
                    self.current_state = State.IDLE

                # 5. --- State Transitions (Single Actions) ---
                if self.current_state != self.prev_state:
                    if self.current_state == State.DRAGGING:
                        pyautogui.mouseDown(button="left")
                    elif self.prev_state == State.DRAGGING:
                        pyautogui.mouseUp(button="left")

                # 6. --- CONTINUOUS ACTIONS ---
                if self.is_mouse_on and right_hand:
                    finger_tip = right_hand[
                        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
                    ]

                    if self.current_state == State.SCROLLING:
                        if self.mouse.is_active:
                            self.mouse.deactivate()
                        scroll_amount = (0.5 - finger_tip.y) * config.SCROLL_SENSITIVITY
                        pyautogui.scroll(int(scroll_amount))
                    elif gestures.is_pointing(right_hand) and self.current_state in [
                        State.IDLE,
                        State.DRAGGING,
                    ]:
                        active_state = (
                            State.DRAGGING
                            if self.current_state == State.DRAGGING
                            else State.MOUSE_MOVEMENT
                        )
                        if not self.mouse.is_active:
                            self.mouse.activate(
                                finger_tip.x * self.screen_width,
                                finger_tip.y * self.screen_height,
                            )
                        delta = self.mouse.update(
                            finger_tip.x * self.screen_width,
                            finger_tip.y * self.screen_height,
                            dt,
                            active_state,
                        )
                        if delta:
                            pyautogui.move(delta[0], delta[1])
                    else:
                        if self.mouse.is_active:
                            self.mouse.deactivate()
                else:
                    if self.mouse.is_active:
                        self.mouse.deactivate()

                # 7. --- Drawing ---
                final_frame = self.ui.draw(
                    frame,
                    self.last_known_result,
                    left_gesture_str,
                    self.current_state,
                    fps,
                    self.is_mouse_on,
                )
                cv.imshow("Virtual Mouse", final_frame)

                if cv.waitKey(1) & 0xFF == 27:  # Exit on ESC
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleans up all resources for a graceful shutdown."""
        logger.info("Shutting down all components...")
        self.settings_gui.stop()
        self.inference_engine.stop()
        self.webcam_stream.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    try:
        app = MouseCamApp()
        app.run()
    except Exception as e:
        logger.error("\n--- AN UNEXPECTED ERROR OCCURRED ---")
        logger.error(traceback.format_exc())
        logger.error("--------------------------------------\n")
        input("Press Enter to exit...")
