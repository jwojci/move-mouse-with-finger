import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

import pyautogui
import cv2 as cv

import config
from mouse import VirtualMouse
from camera import WebcamStream
from vision import Vision

pyautogui.FAILSAFE = False


def draw_feedback(frame, recognition_result, margin, is_active, anchor_point):
    """Draws visual feedback on the frame."""
    # Define colors
    ACTIVE_COLOR = (0, 255, 0)  # Green
    INACTIVE_COLOR = (0, 0, 255)  # Red (BGR)

    landmark_color = ACTIVE_COLOR if is_active else INACTIVE_COLOR

    # Draw the hand landmarks
    if recognition_result.hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils

        for hand_landmarks_list in recognition_result.hand_landmarks:
            proto_landmarks = landmark_pb2.NormalizedLandmarkList()
            proto_landmarks.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks_list
                ]
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=proto_landmarks,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=landmark_color, thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=landmark_color, thickness=2
                ),
            )

    # Draw the gesture name
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        gesture_name = top_gesture.category_name if top_gesture else "None"
        cv.putText(
            frame,
            gesture_name,
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    # Draw the anchor point if the mouse is active
    if is_active and anchor_point:
        anchor_x_px = int(anchor_point[0] * frame.shape[1])
        anchor_y_px = int(anchor_point[1] * frame.shape[0])
        cv.circle(
            frame, (anchor_x_px, anchor_y_px), 10, ACTIVE_COLOR, -1
        )  # Draw a filled circle


def main():
    # --- SETUP ---
    vision = Vision()
    webcam_stream = WebcamStream().start()
    screen_width, screen_height = pyautogui.size()
    mouse = VirtualMouse(screen_width=screen_width, screen_height=screen_height)
    # FPS Counter
    prev_time = 0
    current_time = 0

    # Cooldown for click action to prevent spamming
    click_cooldown = 0.5  # seconds
    last_click_time = 0

    needs_reanchor = True
    # --- MAIN LOOP ---
    while True:
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        frame = webcam_stream.read()
        if frame is None:
            continue

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        recognition_result = vision.recognize(rgb_frame)

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            gesture_name = top_gesture.category_name if top_gesture else "None"
            hand_landmarks = recognition_result.hand_landmarks[0]
            finger_tip = hand_landmarks[
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
            ]

            # If a hand and gesture are detected, process them
            if gesture_name in config.ACTION_GESTURES:
                needs_reanchor = True

                if gesture_name == "Thumb_Up":
                    current_time = time.time()
                    if current_time - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = current_time

            elif gesture_name in config.NEUTRAL_GESTURES:
                if needs_reanchor:
                    mouse.activate(
                        finger_tip.x * screen_width, finger_tip.y * screen_height
                    )
                    needs_reanchor = False
                else:
                    delta = mouse.update(
                        finger_tip.x * screen_width, finger_tip.y * screen_height
                    )
                    if delta:
                        pyautogui.move(delta[0], delta[1])
            draw_feedback(
                frame,
                recognition_result,
                config.ACTIVE_AREA_MARGIN,
                mouse.is_active,
                mouse.anchor_point,
            )
        else:
            # 4. If NO hand is detected, make sure we re-anchor next time
            needs_reanchor = True
        cv.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 70),
            cv.FONT_HERSHEY_PLAIN,
            2,
            (0, 255, 0),
            2,
        )

        cv.imshow("Virtual Mouse", frame)
        if cv.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    # --- CLEANUP ---
    webcam_stream.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
