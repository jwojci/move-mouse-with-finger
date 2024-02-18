import cv2

import mediapipe as mp
import pyautogui

# TODO: Add Left Button click by recognizing gesture of choice
# TODO: Improve stability of the mouse
# TODO: Figure out a way to not lose focus around the corners (Maybe some process the image using opencv? BackgroundSubtractor?)


def main():
    # init mediapipe hands module
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
    # init drawing module
    mp_drawing = mp.solutions.drawing_utils

    video_cap = cv2.VideoCapture(0)

    screen_width, screen_height = pyautogui.size()

    while True:
        ret, frame = video_cap.read()
        frame = cv2.flip(frame, 1)

        results = mp_hands.process(frame)
        try:
            if results.multi_hand_landmarks:  # noqa
                hand_landmarks = results.multi_hand_landmarks[0]

                index_finger_tip = hand_landmarks.landmark[8]
                finger_x, finger_y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(finger_x, finger_y, 0.01)
                # for hand_landmarks in results.multi_hand_landmarks:  # noqa
                #     mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        except Exception as e:
            print(e)
        cv2.imshow('camera_feed', frame)
        k = cv2.waitKey(5)
        if k == 27:
            break

    # hand_landmarker.close()
    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
