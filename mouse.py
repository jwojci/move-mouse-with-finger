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
        self.kf = self._initialize_kalman_filter()
        self.initialized = False
        self.DEADZONE_THRESHOLD = config.DEADZONE_THRESHOLD

    def _initialize_kalman_filter(self):
        """
        Initializes the Kalman filter and its' state variables
        """
        # We have four variables [x, y, vx, vy] but we only measure 2 -> x, y (meaning we only get 2 from MediaPipe)
        kf = KalmanFilter(dim_x=4, dim_z=2)
        # F - State Transition Matrix
        # F - State Transition Matrix
        # How does position and velocity change in one time step (dt = 1 frame)?
        # We use a simple "constant velocity" model:
        # new_position => old_position + velocity
        # new_velocity => old_velocity
        # Now we translate that into matrix rows
        kf.F = np.array(
            [
                [1, 0, 1, 0],  # to calculate new x we need old x and vx
                [0, 1, 0, 1],  # to calculate new y we need old y and vy
                [0, 0, 1, 0],  # to calculate new vx we need old vx
                [0, 0, 0, 1],  # to calculate new vy we need old vy
            ]
        )
        # H - The Measurement Matrix
        # What am I measuring?
        # MediaPipe gives me an x and a y
        # We create a matrix that "picks out" x and y from our state [x, y, vx, vy]
        # Measurement Function H: Maps the 4 state variables to the 2 measured variables.
        kf.H = np.array(
            [
                [1, 0, 0, 0],  # We only measure x
                [0, 1, 0, 0],  # We only measure y
            ]
        )
        # R - Measurement Noise
        # (How much do I trust MediaPipe?)
        # ! This is the most important "knob"
        # Logic: "My hand is perfectly still but the cursor jitters around by a few pixels.
        # This means my MediaPipe measurement has some noise." - R represents this noise
        # Lower R values mean more trust.
        kf.R *= config.KF_MEASUREMENT_NOISE
        # Q - Process Noise Covariance
        # (How much do I trust my own model?)
        # Uncertainty about the process (e.g., unexpected acceleration).
        # Generally we want a very small Q.
        # This tells the filter to mostly trust its physics predictions but be prepared for small, unexpected changes.
        kf.Q *= config.KF_PROCESS_NOISE

        # Initial State Covariance P: Our initial uncertainty about the state. Start high.
        # When I first start the program, I have absolutely no idea where the user's hand is. My initial uncertainty is massive.
        kf.P *= 1000
        return kf

    def update(self, mapped_x, mapped_y):
        """
        Updates the filter with a new measurement and returns smoothed coordinates.
        Returns None if the cursor should not move (i.e., it's in the deadzone).
        """
        # Initialize filter on the first valid frame
        if not self.initialized:
            self.kf.x = np.array([mapped_x, mapped_y, 0, 0])
            self.initialized = True
            return None  # Don't move on the very first frame

        # Kalman Filter steps
        self.kf.predict()
        self.kf.update(np.array([mapped_x, mapped_y]))

        # Apply deadzone
        smoothed_state = self.kf.x
        speed = np.sqrt(smoothed_state[2] ** 2 + smoothed_state[3] ** 2)

        if speed < self.DEADZONE_THRESHOLD:
            return None  # Velocity is too low, stay put

        return int(smoothed_state[0]), int(smoothed_state[1])
