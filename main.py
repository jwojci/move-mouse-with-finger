import mediapipe as mp
import pyautogui
import cv2

import config
from mouse import VirtualMouse
from camera import WebcamStream


def draw_ui(frame, hand_results, margin):
    # Draw the bounding box
    frame_height, frame_width, _ = frame.shape
    start_x = int(margin * frame_width)
    start_y = int(margin * frame_height)
    end_x = frame_width - start_x
    end_y = frame_height - start_y
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    # Draw the hand landmarks
    if hand_results.multi_hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_hand_styles = mp.solutions.drawing_styles
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_hand_styles.get_default_hand_landmarks_style(),
                mp_hand_styles.get_default_hand_connections_style(),
            )


def main():
    # --- SETUP ---
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
    webcam_stream = WebcamStream().start()
    screen_width, screen_height = pyautogui.size()

    mouse = VirtualMouse(screen_width=screen_width, screen_height=screen_height)

    # set the threshold for detecting a "pinch" gesture
    pinch_threshold = config.PINCH_THRESHOLD

    active_area_margin = config.ACTIVE_AREA_MARGIN

    while True:
        frame = webcam_stream.read()  # grab the latest frame from the video stream
        frame = cv2.flip(frame, 1)  # flip the frame for a natural "mirror" view

        # 1. PROCESS: Get hand landmarks from the frame
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)
        frame.flags.writeable = True

        # 2. LOGIC: If a hand is found, update the mouse
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extract and map coordinates
            hand_x = hand_landmarks.landmark[
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
            ].x
            hand_y = hand_landmarks.landmark[
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
            ].y

            # Check if hand is in the active area:
            if (
                active_area_margin < hand_x < 1 - active_area_margin
                and active_area_margin < hand_y < 1 - active_area_margin
            ):
                mapped_x = (
                    (hand_x - active_area_margin)
                    / (1 - 2 * active_area_margin)
                    * screen_width
                )
                mapped_y = (
                    (hand_y - active_area_margin)
                    / (1 - 2 * active_area_margin)
                    * screen_height
                )

                # Update the mouse and get smoothed coordinates
                coords = mouse.update(mapped_x, mapped_y)

                # 3. ACTION: If mouse should move, move it
                if coords:
                    pyautogui.moveTo(coords[0], coords[1])

        # 4. DRAWING: Draw UI elements on the frame
        draw_ui(frame, results, active_area_margin)

        cv2.imshow("camera_feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam_stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
