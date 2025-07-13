import mediapipe as mp
import cv2
import pyautogui
import numpy as np

from loguru import logger

logger.add("func.log")


def get_model(max_num_hands=2):
    return mp.solutions.hands.Hands(max_num_hands)


def get_screen_dim():
    return pyautogui.size()


def process_stream(webcam, model, pinch_threshold):
    """Process webcam stream and perform actions accordingly with hand movement
    Right Index finger moves the mouse, left hand pinch, clicks left mouse button"""


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def process_frame(mp_hands, frame, pinch_threshold, screen_width, screen_height):
    """Processes a single frame to detect hands and control the mouse."""
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        handle_hand_landmarks(
            results, frame, pinch_threshold, screen_width, screen_height
        )


def handle_hand_landmarks(results, frame, pinch_threshold, screen_width, screen_height):
    """
    Iterates through detected hands and applies actions based on handedness.
    - Right hand: Moves the mouse cursor.
    - Left hand: Performs a click on pinch.
    """
    # Initialize drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_hand_styles = mp.solutions.drawing_styles

    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_hand_styles.get_default_hand_landmarks_style(),
            mp_hand_styles.get_default_hand_connections_style(),
        )

        # Get handedness (Left or Right)
        handedness_classification = results.multi_handedness[hand_idx].classification[0]
        handedness = handedness_classification.label

        # Get key landmark points
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
        ]

        # Right Hand: Controls mouse movement
        if handedness == "Right":
            # Map index finger coordinates to screen coordinates
            # Adding a small margin (e.g., 1.2) can help reach screen corners
            cursor_x = int(
                index_finger_tip.x * screen_width * 1.2 - (screen_width * 0.1)
            )
            cursor_y = int(
                index_finger_tip.y * screen_height * 1.2 - (screen_height * 0.1)
            )

            # Move the mouse
            pyautogui.moveTo(cursor_x, cursor_y)

        # Left Hand: Controls clicking
        if handedness == "Left":
            distance = calculate_distance(thumb_tip, index_finger_tip)
            # Check if the thumb and index finger are pinched
            if distance < pinch_threshold:
                pyautogui.click(button="left")
