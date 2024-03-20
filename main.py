import cv2

import func


# TODO: Figure out a way to not lose focus around the corners


def main():
    # init mediapipe hands module
    mp_hands = func.get_model()
    # init drawing module
    # mp_drawing = mp.solutions.drawing_utils

    webcam = func.get_webcam()

    screen_width, screen_height = func.get_screen_dim()

    pinch_threshold = 0.09

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # frame = cv2.GaussianBlur(frame, (7, 7), 0)
        func.process_frame(mp_hands, frame, pinch_threshold, screen_width, screen_height)

        cv2.imshow('camera_feed', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
