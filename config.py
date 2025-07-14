# Screen and Camera settings
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# Mouse Control Settings
ACTIVE_AREA_MARGIN = 0.05  # Percentage of the screen to use as a deadzone border
DEADZONE_THRESHOLD = 2  # Velocity threshold to consider the mouse "stopped"
VIRTUAL_TOUCHPAD_SIZE = 0.2
ACCELERATION_FACTOR = 0.125

# Kalman Filter Tuning
KF_MEASUREMENT_NOISE = 0.6  # R value - Trust in MediaPipe measurements
KF_PROCESS_NOISE = 0.07  # Q value - Trust in the physics model

# Gesture Settings (we'll use this later)
PINCH_THRESHOLD = 0.04

NEUTRAL_GESTURES = {"Open_Palm", "Pointing_Up", "None"}
ACTION_GESTURES = {"Thumb_Up", "Closed_Fist", "Victory"}
