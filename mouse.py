import numpy as np
from filterpy.kalman import KalmanFilter

import config
from utils import State


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

    def update(self, finger_x_px, finger_y_px, dt, state):
        """
        Updates the mouse position based on the new finger coordinates and delta time.
        Returns the (dx, dy) mouse movement delta.
        """
        if not self.is_active or not self.initialized or dt <= 0:
            return None

        # 1. Correct the Kalman Filter's physics model with the real dt
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt

        # 2. Kalman Filter: Predict and Update
        self.kf.predict()
        self.kf.update(np.array([finger_x_px, finger_y_px]))
        smoothed_pos = self.kf.x[:2]

        # 3. Calculate Relative Delta
        delta_pos = smoothed_pos - self.last_smoothed_pos
        self.last_smoothed_pos = smoothed_pos

        # 4. Deadzone
        if np.linalg.norm(delta_pos) < config.DEADZONE_THRESHOLD:
            return None

        # 5. Apply Sensitivity and Acceleration
        speed = np.linalg.norm(delta_pos) / dt  # Velocity in pixels/sec
        movement_multiplier = 1.0 + (speed * config.ACCELERATION_FACTOR)

        if state == State.DRAGGING:
            movement_multiplier /= 2

        # Scale the delta by our sensitivity factor
        final_delta = delta_pos * config.SENSITIVITY * movement_multiplier

        return (final_delta[0], final_delta[1])
