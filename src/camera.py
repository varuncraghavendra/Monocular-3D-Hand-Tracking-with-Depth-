# Varun Raghavendra
# PRCV Spring 2026
# OpenCV VideoCapture wrapper that always returns the latest frame

import cv2


class ThreadedCamera:

    def __init__(self, src=0, width=960, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    def read(self):
        # Reads one frame from the capture device, returning None on failure.
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        # Releases the underlying VideoCapture handle, suppressing any errors.
        try:
            self.cap.release()
        except Exception:
            pass
