from threading import Thread

import cv2

import func

# TODO: Figure out a way to not lose focus around the corners


class WebcamStream:
    """
    A class to read frames from a webcam in a dedicated thread.
    This prevents the main processing loop from blocking while waiting for a new frame.
    """

    def __init__(self, src=0):
        # Initialize the video capture stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Read the first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Flag to indicate if the thread should be stopped
        self.stopped = False

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
        # Indicate that the thread should stop
        self.stopped = True


def main():
    # init mediapipe hands module
    mp_hands = func.get_model()

    # start webcam stream on a seperate thread
    webcam_stream = WebcamStream().start()

    # get screen dimensions for mouse control
    screen_width, screen_height = func.get_screen_dim()

    # set the threshold for detecting a "pinch" gesture
    pinch_threshold = 0.09

    while True:
        # grab the latest frame from the video stream
        frame = webcam_stream.read()

        # flip the frame for a natural "mirror" view
        frame = cv2.flip(frame, 1)

        # process the frame for hand tracking and mouse control
        func.process_frame(
            mp_hands, frame, pinch_threshold, screen_width, screen_height
        )

        # display the resulting frame
        cv2.imshow("camera_feed", frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    webcam_stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
