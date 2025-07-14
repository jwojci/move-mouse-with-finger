import numpy as np
from filterpy.kalman import KalmanFilter

import config


class VirtualMouse:
    """
    Manages the state and logic for the virtual mouse, including
    Kalman filter smoothing and a velocity-based deadzone.
    """

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.is_active = False
        self.anchor_point = None
        self.sensitivity = 1.5

        self.kf = self._initialize_kalman_filter()
        self.initialized = False
        self.DEADZONE_THRESHOLD = config.DEADZONE_THRESHOLD

    def _initialize_kalman_filter(self):
        """
        Initializes the Kalman filter and its' state variables
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        kf.R *= config.KF_MEASUREMENT_NOISE
        kf.Q *= config.KF_PROCESS_NOISE
        kf.P *= 1000
        return kf

    def activate(self, anchor_x, anchor_y):
        self.is_active = True
        self.anchor_point = (anchor_x, anchor_y)
        self.kf.x = np.array([anchor_x, anchor_y, 0, 0])
        self.initialized = True

    def deactivate(self):
        self.is_active = False
        self.initialized = False

    def update(self, mapped_x, mapped_y):
        """
        The main update loop.
        - Smooths the raw input using the Kalman Filter.
        - Checks for jitter using a velocity deadzone.
        - Calculates relative movement with dynamic sensitivity.
        """
        if not self.is_active or not self.initialized:
            return None

        # --- 1. Kalman Filter Smoothing
        # Predict the next state and update with the new measurement
        self.kf.predict()
        self.kf.update(np.array([[mapped_x], [mapped_y]]))
        smoothed_state = self.kf.x

        # --- 2. Velocity Deadzone
        speed = np.sqrt(smoothed_state[2] ** 2 + smoothed_state[3] ** 2)
        if speed < config.DEADZONE_THRESHOLD:
            return None

        # --- 3. Dynamic Sensitivity
        # Calculate the distance from anchor to determine sensitivity
        distance_from_anchor = np.sqrt(
            (mapped_x - self.anchor_point[0]) ** 2
            + (mapped_y - self.anchor_point[1]) ** 2
        )

        # Define sensitivity zones
        if distance_from_anchor < 30:
            sensitivity = 0.2
        elif distance_from_anchor < 100:
            sensitivity = 5
        else:
            sensitivity = 4

        # 4. Calculate relative deltas
        delta_x = smoothed_state[2] * sensitivity
        delta_y = smoothed_state[3] * sensitivity

        return (delta_x, delta_y, distance_from_anchor)
