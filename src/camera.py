import cv2


class ThreadedCamera:
    # Thin OpenCV VideoCapture wrapper with buffer=1 so read() always returns
    # the latest frame rather than a stale one from the queue.

    def __init__(self, src=0, width=960, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
