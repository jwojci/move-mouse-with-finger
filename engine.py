import threading
import queue

from vision import Vision


class InferenceEngine:
    """
    Manages running gesture recognition in a separate thread using a
    producer-consumer queue for efficient, non-blocking communication.
    """

    def __init__(self):
        self.vision = Vision()
        self.latest_result = None
        self.lock = threading.Lock()

        # A queue to hold frames for the inference thread to process.
        # maxsize=1 ensures we only process the most recent frame and drop old ones.
        self.frame_queue = queue.Queue(maxsize=1)

        self.is_running = False
        self.thread = threading.Thread(target=self._run_inference, daemon=True)

    def _run_inference(self):
        """The main loop for the background inference thread."""
        while self.is_running:
            try:
                # This call BLOCKS until a frame is available in the queue.
                # The thread sleeps efficiently here, consuming no CPU.
                frame_to_process = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue  # If no frame in 1s, continue waiting

            if frame_to_process is None:  # Shutdown signal
                continue

            # Run the expensive recognition
            result = self.vision.recognize(frame_to_process)

            # Use the lock only for this brief update
            with self.lock:
                self.latest_result = result

            # Inform the queue that the task is done
            self.frame_queue.task_done()

    def start(self):
        """Starts the background inference thread."""
        if not self.is_running:
            self.is_running = True
            self.thread.start()

    def stop(self):
        """Stops the background inference thread cleanly."""
        if self.is_running:
            return
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join()  # Wait for the thread to finish its loop

    def update_frame(self, frame):
        """Receives a new frame from the main thread for processing."""
        # If the queue is full, it means the model is still busy.
        # We first clear the queue to drop the stale frame
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        # then we put the new, most recent frame into the queue.
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # If it's still full (rare), we just drop this frame

    def get_latest_result(self):
        """Returns the most recent recognition result."""
        with self.lock:
            return self.latest_result
