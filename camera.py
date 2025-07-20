import time
from threading import Thread

import cv2

import config


class WebcamStream:
    """
    A class to read frames from a webcam in a dedicated thread.
    This prevents the main processing loop from blocking while waiting for a new frame.
    """

    def __init__(self, src=0):
        # Initialize the video capture stream
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError("Cannot open webcam")
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

        # Read the first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Flag to indicate if the thread should be stopped
        self.stopped = False
        self.thread = None

    def start(self):
        # Start a thread to read frames from the video stream
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        # Loop until the thread is stopped
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        """Stops the thread and releases the webcam resource."""
        # Indicate that the thread should stop
        self.stopped = True
        # Wait for the thread to finish
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        # Release the video stream resource
        self.stream.release()
