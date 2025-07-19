# Screen and Camera settings
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# Processing resolution (for model inference)
PROCESSING_WIDTH = 320
PROCESSING_HEIGHT = 240

# Mouse Control Settings
SENSITIVITY = 0.9
ACTIVE_AREA_MARGIN = 0.05  # Percentage of the screen to use as a deadzone border
DEADZONE_THRESHOLD = 1.5  # Velocity threshold to consider the mouse "stopped"
VIRTUAL_TOUCHPAD_SIZE = 0.2
ACCELERATION_FACTOR = 0.02

# Kalman Filter Tuning
# If the cursor is too jittery increase this value, if it feels heavy or laggy decrease it
KF_MEASUREMENT_NOISE = 0.3  # R value - Trust in MediaPipe measurements
# If the cursor feels slow to react increas this value, if it overshoots or feels wobbly decrease it
KF_PROCESS_NOISE = 0.01  # Q value - Trust in the physics model

# Gesture Settings (we'll use this later)
PINCH_THRESHOLD = 0.04

NEUTRAL_GESTURES = {"Open_Palm", "Pointing_Up", "None"}
ACTION_GESTURES = {"Thumb_Up", "Closed_Fist", "Victory"}
