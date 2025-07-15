import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from loguru import logger


class Vision:
    def __init__(self, model_path="model/gesture_recognizer.task"):
        try:
            self.base_options = python.BaseOptions(model_asset_path=model_path)
            self.recognizer_options = vision.GestureRecognizerOptions(
                base_options=self.base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(
                self.recognizer_options
            )
        except Exception as e:
            logger.error(f"Error during Vision Class initialization: {e}")
            raise Exception(e)

    def recognize(self, frame):
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)
            # Recognize gesture
            return self.recognizer.recognize_for_video(mp_image, timestamp_ms)
        except Exception as e:
            logger.error(f"Error while trying to run the recognition: {e}")
            raise Exception(e)
