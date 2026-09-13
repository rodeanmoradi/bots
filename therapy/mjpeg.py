"""Read an HTTP MJPEG stream (DroidCam, IP Webcam) through a cv2.VideoCapture-like interface."""

import threading
import time
import urllib.request

import cv2
import numpy as np


class MjpegStream:
    """cv2.VideoCapture-like reader for HTTP MJPEG streams such as DroidCam or IP Webcam."""

    def __init__(self, url, timeout=5.0):
        self.url = url
        self.last_error = None
        self._timeout = timeout
        self._frame = None
        self._shape = (0, 0)
        self._running = True
        self._new_frame = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        # JPEGs are cut out of the raw bytes by their start/end markers, so any multipart
        # boundary format works; a dropped connection is retried below.
        while self._running:
            try:
                with urllib.request.urlopen(self.url, timeout=self._timeout) as stream:
                    buf = b""
                    while self._running:
                        chunk = stream.read1(65536)
                        if not chunk:
                            break
                        buf += chunk
                        end = buf.rfind(b"\xff\xd9")
                        start = buf.rfind(b"\xff\xd8", 0, end) if end != -1 else -1
                        if start == -1:
                            if len(buf) > 8_000_000:
                                buf = b""
                            continue
                        jpeg, buf = buf[start:end + 2], buf[end + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._new_frame:
                                self._frame = frame
                                self._shape = frame.shape[:2]
                                self._new_frame.notify_all()
            except OSError as e:
                self.last_error = e
            if self._running:
                time.sleep(1.0)

    def read(self):
        # Only the newest frame is kept, so a slow consumer skips frames instead of lagging.
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._frame is not None or not self._running,
                                     timeout=self._timeout)
            frame, self._frame = self._frame, None
        return frame is not None, frame

    def isOpened(self):
        return self._running

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._shape[0])
        return 0.0

    def set(self, prop, value):
        return False

    def release(self):
        with self._new_frame:
            self._running = False
            self._new_frame.notify_all()
