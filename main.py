import cv2

import mediapipe as mp
import pyautogui
import numpy as np
import sys


# TODO: Figure out a way to not lose focus around the corners

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def main():
    # init mediapipe hands module
    mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
    # init drawing module
    mp_drawing = mp.solutions.drawing_utils

    video_cap = cv2.VideoCapture(0)

    screen_width, screen_height = pyautogui.size()

    pinch_threshold = 0.09

    while True:
        ret, frame = video_cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        results = mp_hands.process(frame)
        try:
            if results.multi_hand_landmarks:  # noqa
                # For identifying if the hand is right or left
                handedness = results.multi_handedness[0].classification[0].label

                hand_landmarks = results.multi_hand_landmarks[0]

                thumb_tip = hand_landmarks.landmark[4]
                index_finger_tip = hand_landmarks.landmark[8]

                index_finger_x, index_finger_y = int(index_finger_tip.x * screen_width), int(
                    index_finger_tip.y * screen_height)

                distance = calculate_distance(thumb_tip, index_finger_tip)

                if handedness == "Right":
                    pyautogui.moveTo(index_finger_x, index_finger_y)

                if handedness == "Left" and distance < pinch_threshold:
                    pyautogui.click(button="left")

                # for hand_landmarks in results.multi_hand_landmarks:  # noqa
                #     mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        except Exception as e:
            print(e)
        cv2.imshow('camera_feed', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    # hand_landmarker.close()
    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
