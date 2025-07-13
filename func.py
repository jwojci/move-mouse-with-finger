import mediapipe as mp
import cv2
import pyautogui
import numpy as np
from filterpy.kalman import KalmanFilter
from loguru import logger

logger.add("func.log")

# --- Kalman Filter Setup ---
# We have four variables [x, y, vx, vy] but we only measure 2 -> x, y (meaning we only get 2 from MediaPipe)
kf = KalmanFilter(dim_x=4, dim_z=2)

# F - State Transition Matrix
# How does position and velocity change in one time step (dt = 1 frame)?
# We use a simple "constant velocity" model:
# new_position => old_position + velocity
# new_velocity => old_velocity
# Now we translate that into matrix rows
kf.F = np.array(
    [
        [1, 0, 1, 0],  # 1*x + 0*y + 1*vx + 0*vy
        [0, 1, 0, 1],  # 0*x + 1*y + 0*vx + 1*vy
        [0, 0, 1, 0],  # 0*x + 0*y + 1*vx + 0*vy
        [0, 0, 0, 1],  # 0*x + 0*y + 0*vx + 1*vy
    ]
)

# H - The Measurement Matrix
# What am I measuring?
# MediaPipe gives me an x and a y
# We create a matrix that "picks out" x and y from our state [x, y, vx, vy]
# Measurement Function H: Maps the 4 state variables to the 2 measured variables.
kf.H = np.array(
    [
        [1, 0, 0, 0],  # We only measure x
        [0, 1, 0, 0],  # We only measure y
    ]
)

# R - Measurement Noise
# (How much do I trust MediaPipe?)
# ! This is the most important "knob"
# Logic: "My hand is perfectly still but the cursor jitters around by a few pixels.
# This means my MediaPipe measurement has some noise." - R represents this noise
# Lower R values mean more trust.
kf.R = np.array(
    [
        [1, 0],  # How much we trust our x measurement?
        [0, 1],  # How much we trust our y measurement?
    ]
)

# Q - Process Noise Covariance
# (How much do I trust my own model?)
# Uncertainty about the process (e.g., unexpected acceleration).
# Generally we want a very small Q.
# This tells the filter to mostly trust its physics predictions but be prepared for small, unexpected changes.
kf.Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) * 0.07

# Initial State Covariance P: Our initial uncertainty about the state. Start high.
# When I first start the program, I have absolutely no idea where the user's hand is. My initial uncertainty is massive.
kf.P *= 1000

# Initial state [x, y, vx, vy]
kf.x = np.array([0, 0, 0, 0])


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
        handedness = results.multi_handedness[hand_idx].classification[0].label

        # Get key landmark points
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
        ]

        # Right Hand: Controls mouse movement
        if handedness == "Right":
            # --- Kalman Filter integration ---
            # 1. Predict the next state
            kf.predict()

            # 2. Get the new measurement (the noisy finger position)
            measured_x = index_finger_tip.x * screen_width
            measured_y = index_finger_tip.y * screen_height
            measurement = np.array([[measured_x], [measured_y]])

            # 3. Update the filter with the new measurement
            kf.update(measurement)

            # Get the smoothed state (the filter's estimate)
            smoothed_state = kf.x
            cursor_x = int(smoothed_state[0])
            cursor_y = int(smoothed_state[1])

            # Map index finger coordinates to screen coordinates
            # Adding a small margin (e.g., 1.2) can help reach screen corners
            # cursor_x = int(
            #     index_finger_tip.x * screen_width * 1.2 - (screen_width * 0.1)
            # )
            # cursor_y = int(
            #     index_finger_tip.y * screen_height * 1.2 - (screen_height * 0.1)
            # )

            # Move the mouse
            pyautogui.moveTo(cursor_x, cursor_y)

        # Left Hand: Controls clicking
        if handedness == "Left":
            distance = calculate_distance(thumb_tip, index_finger_tip)
            # Check if the thumb and index finger are pinched
            if distance < pinch_threshold:
                pyautogui.click(button="left")
