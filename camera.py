import cv2
import threading
import time

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS
)


class Camera:

    def __init__(self):

        self.cap = None

        self.frame = None

        self.running = False

        self.lock = threading.Lock()

        self.thread = None


    def start(self):

        if self.running:
            return

        # بدون CAP_DSHOW
        self.cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

        if not self.cap.isOpened():

            raise Exception(
                "Cannot open camera"
            )


        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            FPS
        )


        self.running = True


        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()



    def update(self):

        while self.running:

            ret, frame = self.cap.read()


            if ret:

                with self.lock:

                    self.frame = frame.copy()


            time.sleep(0.01)



    def read(self):

        with self.lock:

            if self.frame is None:

                return None

            return self.frame.copy()



    def stop(self):

        self.running = False


        if self.thread:

            self.thread.join(
                timeout=1
            )


        if self.cap:

            self.cap.release()

            self.cap = None