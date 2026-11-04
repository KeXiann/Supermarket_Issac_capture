"""Bounded asynchronous video writer with a joined encoder on episode save."""

import os
import queue
import threading

import av
import numpy as np


class VideoWriter:
    def __init__(self, output_path, width, height, fps, codec="h264", buffer_size=50):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self._error = None
        self._stopped = False
        self._thread = threading.Thread(target=self._writer_worker, daemon=True)
        self._thread.start()

    def _assert_dimensions(self, frame: np.ndarray):
        if frame.shape != (self.stream.height, self.stream.width, 3):
            raise ValueError(f"Incorrect frame dimensions: {frame.shape}; expected "
                             f"{self.stream.width}x{self.stream.height} RGB")

    def _put(self, item):
        while True:
            if self._error is not None:
                raise RuntimeError(f"Video encoder failed: {self.output_path}") from self._error
            try:
                self.queue.put(item, timeout=.1)
                return
            except queue.Full:
                continue

    def add_frame(self, frame):
        if self._stopped:
            raise RuntimeError("Video writer already stopped")
        self._assert_dimensions(frame)
        self._put(frame.copy())

    def _writer_worker(self):
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in self.stream.encode(video_frame):
                    self.container.mux(packet)
            for packet in self.stream.encode():
                self.container.mux(packet)
        except BaseException as error:
            self._error = error
        finally:
            try:
                self.container.close()
            except BaseException as error:
                self._error = self._error or error

    def stop(self):
        # A queue can be empty while the last frame is still encoding. A sentinel
        # and join ensure all frames finish before flushing/closing, on one thread.
        if not self._stopped:
            self._stopped = True
            try:
                self._put(None)
            finally:
                self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Video encoder failed: {self.output_path}") from self._error
        return self.output_path

    def cancel(self):
        try:
            self.stop()
        finally:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
