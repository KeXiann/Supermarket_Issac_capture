"""Decode real H264 after TCP backpressure and callback reconnection.

These use real local sockets with constrained buffers, so a receiver pause
can reproduce the old one-second sendall timeout without a real headset.
"""
import io
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_video_server import CameraRequest, VideoSession


def read_exact(sock, length):
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise EOFError("video socket closed mid-frame")
        data.extend(chunk)
    return bytes(data)


def decode_one(sock, decoder):
    size, = struct.unpack(">I", read_exact(sock, 4))
    if not 0 < size < 4_000_000:
        raise AssertionError(f"frame prefix corrupted: {size}")
    packet = read_exact(sock, size)
    return decoder.decode(av.Packet(packet))


class ChangingImage:
    def __init__(self):
        rng = np.random.default_rng(20261027)
        self.images = []
        for _ in range(12):
            pixels = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="JPEG", quality=90)
            self.images.append(stream.getvalue())

    def snapshot(self):
        count = int(time.monotonic() * 30)
        return (self.images[count % len(self.images)], time.time(), "bgr"), count


class NativeTransportTests(unittest.TestCase):
    def make_session(self, listener, source, status, stall=5.):
        req = CameraRequest(640, 360, 30, 4_000_000, 0, 2,
                            listener.getsockname()[1], "ZED", "127.0.0.1")
        return VideoSession(req, "127.0.0.1", source, str(status),
                            max_bitrate=4_000_000, send_stall_seconds=stall,
                            send_buffer_bytes=4096)

    def test_two_second_receiver_pause_preserves_framing_and_decoding(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
            listener.bind(("127.0.0.1", 0))
            listener.listen(4)
            listener.settimeout(6)
            session = self.make_session(listener, ChangingImage(), Path(temporary) / "status.json")
            try:
                video, _ = listener.accept()
                with video:
                    video.settimeout(8)
                    decoder = av.CodecContext.create("h264", "r")
                    initial = decode_one(video, decoder)
                    self.assertEqual(initial[0].width, 640)
                    time.sleep(2.)  # Old sendall(timeout=1) killed this stream.
                    decoded = []
                    for _ in range(20):
                        decoded.extend(decode_one(video, decoder))
                    self.assertGreaterEqual(len(decoded), 20)
                    self.assertGreater(session.backpressure_events, 0)
                    self.assertEqual(session.reconnects, 0)
                    self.assertGreater(np.abs(decoded[-1].to_ndarray(format="rgb24").astype(float)
                        - decoded[0].to_ndarray(format="rgb24").astype(float)).mean(), 8.)
            finally:
                session.close()
            self.assertFalse(session.thread.is_alive())

    def test_connection_reset_reconnects_without_a_new_open_request(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(4)
            listener.settimeout(6)
            status = Path(temporary) / "status.json"
            session = self.make_session(listener, ChangingImage(), status)
            try:
                first, _ = listener.accept()
                first.settimeout(5)
                self.assertTrue(decode_one(first, av.CodecContext.create("h264", "r")))
                first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                first.close()
                second, _ = listener.accept()
                with second:
                    second.settimeout(5)
                    # A new decoder must decode immediately: fresh SPS/PPS + IDR.
                    decoded = decode_one(second, av.CodecContext.create("h264", "r"))
                    self.assertEqual(decoded[0].height, 360)
                    self.assertGreaterEqual(session.reconnects, 1)
            finally:
                session.close()
            state = json.loads(status.read_text())
            self.assertFalse(state["connected"])
            self.assertTrue(state["source_live"])
            self.assertEqual(state["transport_state"], "stopped")

    def test_permanently_stalled_receiver_is_reconnected_and_close_is_cancellable(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            listener.bind(("127.0.0.1", 0))
            listener.listen(4)
            listener.settimeout(6)
            session = self.make_session(listener, ChangingImage(), Path(temporary) / "status.json", stall=.5)
            first = second = None
            try:
                first, _ = listener.accept()
                # Do not consume even the first access unit: force a true stall.
                second, _ = listener.accept()
                second.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1_048_576)
                second.settimeout(5)
                self.assertTrue(decode_one(second, av.CodecContext.create("h264", "r")))
                self.assertIn("TimeoutError", session.last_error)
                self.assertGreaterEqual(session.reconnects, 1)
                start = time.monotonic()
                session.close()
                self.assertLess(time.monotonic() - start, 2.)
            finally:
                session.close()
                for sock in (first, second):
                    if sock is not None:
                        sock.close()
            self.assertFalse(session.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
