import mediapipe as mp
import cv2
import pyautogui
import numpy as np


def get_model(max_num_hands=2):
    return mp.solutions.hands.Hands(max_num_hands)


def get_webcam():
    return cv2.VideoCapture(0)


def get_screen_dim():
    return pyautogui.size()


def process_stream(webcam, model, pinch_threshold):
    """Process webcam stream and perform actions accordingly with hand movement
       Right Index finger moves the mouse, left hand pinch, clicks left mouse button"""


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def process_frame(mp_hands, frame, pinch_threshold, screen_width, screen_height):
    results = mp_hands.process(frame)
    if results.multi_hand_landmarks:
        handle_hand_landmarks(results, pinch_threshold, screen_width, screen_height)


def handle_hand_landmarks(results, pinch_threshold, screen_width, screen_height):
    handedness = results.multi_handedness[0].classification[0].label
    hand_landmarks = results.multi_hand_landmarks[0]

    thumb_tip = hand_landmarks.landmark[4]
    index_finger_tip = hand_landmarks.landmark[8]

    index_finger_x, index_finger_y = int(index_finger_tip.x * screen_width), int(
        index_finger_tip.y * screen_height)

    distance = calculate_distance(thumb_tip, index_finger_tip)
    # print(distance)
    if handedness == "Right":
        pyautogui.moveTo(index_finger_x, index_finger_y)

    if handedness == "Left" and pinch_threshold > distance > 0.03:
        pyautogui.click(button="left")

    # for hand_landmarks in results.multi_hand_landmarks:  # noqa
    #     mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
