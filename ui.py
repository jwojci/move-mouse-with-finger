import cv2 as cv
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


class UserInterface:
    """Handles all drawing operations for the application's UI."""

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.ACTIVE_COLOR = (0, 255, 0)  # Green
        self.INACTIVE_COLOR = (0, 0, 255)  # Red
        self.TEXT_COLOR = (255, 255, 255)  # White

    def draw(self, frame, recognition_result, gesture_name, is_active, fps):
        """The main drawing method. Call this once per frame."""
        landmark_color = self.ACTIVE_COLOR if is_active else self.INACTIVE_COLOR

        # Draw hand landmarks if a hand is detected
        if recognition_result and recognition_result.hand_landmarks:
            self._draw_hand_landmarks(
                frame, recognition_result.hand_landmarks, landmark_color
            )

        # Draw the current gesture name
        self._draw_text(frame, f"Gesture: {gesture_name}", (50, 50))

        # Draw the FPS counter
        self._draw_text(frame, f"FPS: {int(fps)}", (10, 90))

        return frame

    def _draw_hand_landmarks(self, frame, hand_landmarks_lists, color):
        """Draws the hand skeleton."""
        for hand_landmarks_list in hand_landmarks_lists:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks_list
                ]
            )
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks_proto,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=color, thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=color, thickness=2
                ),
            )

    def _draw_text(self, frame, text, position):
        """Utility to draw text on the frame."""
        cv.putText(
            frame,
            text,
            position,
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            self.TEXT_COLOR,
            2,
            cv.LINE_AA,
        )
