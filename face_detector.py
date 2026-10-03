import cv2

from config import (
    YUNET_MODEL,
    FRAME_WIDTH,
    FRAME_HEIGHT
)



class FaceDetector:


    def __init__(self):

        self.detector = cv2.FaceDetectorYN.create(

            YUNET_MODEL,

            "",

            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            ),

            0.9,

            0.3,

            5000

        )


    def detect(self, frame):


        self.detector.setInputSize(
            (
                frame.shape[1],
                frame.shape[0]
            )
        )


        _, faces = self.detector.detect(
            frame
        )


        if faces is None:

            return []


        return faces