import numpy as np
from filterpy.kalman import KalmanFilter

import config


class VirtualMouse:
    """
    Manages the state and logic for the virtual mouse.
    Uses a Kalman filter to smooth hand tracking data and calculates relative
    movement for intuitive cursor control.
    """

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.is_active = False
        self.last_smoothed_pos = None
        self.kf = self._initialize_kalman_filter()
        self.initialized = False

    def _initialize_kalman_filter(self):
        """Initializes the Kalman filter for smoothing 2D position and velocity."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        # State transition matrix (F)
        # Assumes constant velocity model
        dt = 1.0  # time step
        kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        # Measurement matrix (H)
        # We only measure position
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        # Measurement noise covariance (R) - Trust in MediaPipe's raw (x, y)
        kf.R *= config.KF_MEASUREMENT_NOISE
        # Process noise covariance (Q) - How much we expect the hand's velocity to change
        kf.Q = np.eye(4) * config.KF_PROCESS_NOISE
        # Initial state covariance (P)
        kf.P *= 100
        return kf

    def activate(self, start_x_px, start_y_px):
        """Activates mouse control and initializes the filter's state."""
        self.is_active = True
        # Initialize filter at the starting position with zero velocity
        self.kf.x = np.array([start_x_px, start_y_px, 0, 0])
        self.last_smoothed_pos = self.kf.x[:2]  # Store initial smoothed position
        self.initialized = True
        print("Mouse control ACTIVATED")

    def deactivate(self):
        """Deactivates mouse control."""
        self.is_active = False
        self.initialized = False
        self.last_smoothed_pos = None
        print("Mouse control DEACTIVATED")

    def update(self, finger_x_px, finger_y_px):
        """
        Updates the mouse position based on the new finger coordinates.
        Returns the (dx, dy) mouse movement delta.
        """
        if not self.is_active or not self.initialized:
            return None

        # 1. Kalman Filter: Predict next state and update with new measurement
        self.kf.predict()
        self.kf.update(np.array([finger_x_px, finger_y_px]))
        smoothed_pos = self.kf.x[:2]  # Extract smoothed (x, y) position

        # 2. Calculate Relative Delta
        # movement is the change in position
        delta_pos = smoothed_pos - self.last_smoothed_pos

        # 3. Update State for Next Frame
        # The current smoothed position becomes the last position for the next iteration
        self.last_smoothed_pos = smoothed_pos

        # 4. Deadzone: If the hand moved very little, ignore to prevent drift
        if np.linalg.norm(delta_pos) < config.DEADZONE_THRESHOLD:
            return None

        # 5. Optional: Acceleration
        # Apply a multiplier for larger/faster movements. Keep the factor small.
        speed = np.linalg.norm(delta_pos)
        movement_multiplier = 1.0 + (speed * config.ACCELERATION_FACTOR)
        final_delta = delta_pos * movement_multiplier

        return (final_delta[0], final_delta[1])
