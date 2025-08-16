import numpy as np
import mediapipe as mp

import config

# To access the HandLandmark enum
HandLandmark = mp.solutions.hands.HandLandmark


def get_finger_distance(hand_landmarks, finger1, finger2):
    """Calculates the Euclidean distance between two finger landmarks."""
    p1 = hand_landmarks[finger1]
    p2 = hand_landmarks[finger2]
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))


def is_pointing(hand_landmarks):
    """Checks if the hand is in a pointing gesture."""
    index_tip_y = hand_landmarks[HandLandmark.INDEX_FINGER_TIP].y
    middle_mcp_y = hand_landmarks[HandLandmark.MIDDLE_FINGER_MCP].y
    return (middle_mcp_y - index_tip_y) > config.POINTING_THRESHOLD


def is_fisted(hand_landmarks):
    """Checks if the hand is in a fist gesture by measuring distance from index tip to wrist."""
    index_tip = hand_landmarks[HandLandmark.INDEX_FINGER_TIP]
    wrist = hand_landmarks[HandLandmark.WRIST]
    distance = np.linalg.norm(
        np.array([index_tip.x, index_tip.y]) - np.array([wrist.x, wrist.y])
    )
    return distance < config.FIST_THRESHOLD


def is_pinched(hand_landmarks):
    """Checks if the thumb and index finger are pinched together."""
    return (
        get_finger_distance(
            hand_landmarks,
            HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.THUMB_TIP,
        )
        < config.PINCH_THRESHOLD
    )
