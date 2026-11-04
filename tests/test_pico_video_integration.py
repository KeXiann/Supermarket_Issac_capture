"""Exercise the native headset wire protocol and decode its actual H264 output.

Run with .venv_ego_video/bin/python -m unittest discover -s tests
    -p test_pico_video_integration.py
"""
import base64
import io
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import av
import msgpack
import numpy as np
from PIL import Image
import zmq

ROOT = Path(__file__).resolve().parents[1]


def exact(sock, length):
    result = bytearray()
    while len(result) < length:
        chunk = sock.recv(length - len(result))
        if not chunk:
            raise EOFError("video socket closed")
        result.extend(chunk)
    return bytes(result)


def control_packet(command, payload=b""):
    cmd = command.encode()
    body = struct.pack("<i", len(cmd)) + cmd + struct.pack("<i", len(payload)) + payload
    return struct.pack(">I", len(body)) + body


class NativeVideoIntegration(unittest.TestCase):
    def test_camera_request_decode_staleness_and_reconnect(self):
        with tempfile.TemporaryDirectory() as temporary, zmq.Context() as context:
            pub = context.socket(zmq.PUB)
            port = pub.bind_to_random_port("tcp://127.0.0.1")
            with socket.socket() as free_port:
                free_port.bind(("127.0.0.1", 0))
                control_port = free_port.getsockname()[1]
            logfile = open(Path(temporary) / "server.log", "w+")
            process = subprocess.Popen([sys.executable, str(ROOT / "bridge/pico_video_server.py"),
                "--host", "127.0.0.1", "--port", str(control_port),
                "--source", f"tcp://127.0.0.1:{port}", "--stale-seconds", "0.6"],
                stdout=logfile, stderr=subprocess.STDOUT)
            publishing = threading.Event()
            publishing.set()
            def publish():
                frame = 0
                while publishing.is_set():
                    image = Image.new("RGB", (160, 120), (220, frame % 150, 15))
                    data = io.BytesIO()
                    image.save(data, format="JPEG")
                    pub.send(msgpack.packb({"timestamps": {"ego_view": time.time()},
                        "images": {"ego_view": base64.b64encode(data.getvalue()).decode()},
                        "jpeg_encoder_input": "bgr"}, use_bin_type=True))
                    frame += 8
                    publishing.wait(.001)
                    time.sleep(.03)
            publisher = threading.Thread(target=publish)
            publisher.start()
            try:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        ctrl = socket.create_connection(("127.0.0.1", control_port), timeout=.5)
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            self.fail("video server did not start")
                        time.sleep(.05)
                # Real headset request dimensions, with fragmented headers and
                # coalesced commands to exercise TCP's lack of message boundaries.
                with ctrl, socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    listener.listen(2)
                    listener.settimeout(5)
                    request = b"\xca\xfe\x01" + struct.pack("<7i", 2560, 720, 60,
                        4_000_000, 0, 2, listener.getsockname()[1]) + b"\x03ZED\x09127.0.0.1"
                    packet = control_packet("OPEN_CAMERA", request)
                    ctrl.sendall(control_packet("AUDIO_SESSION", b"0123456789abcdef0123456789abcdef") + packet[:2])
                    ctrl.sendall(packet[2:7])
                    ctrl.sendall(packet[7:])
                    ack_size, = struct.unpack(">I", exact(ctrl, 4))
                    ack = exact(ctrl, ack_size)
                    self.assertIn(b"AUDIO_CONFIG", ack)
                    cmd_size, = struct.unpack_from("<I", ack)
                    metadata = json.loads(ack[8 + cmd_size:])
                    self.assertEqual(metadata["video_stereo_layout"], "side_by_side")
                    self.assertEqual(metadata["audio_stream_port"], 0)
                    video, _ = listener.accept()
                    decoder = av.CodecContext.create("h264", "r")
                    images = []
                    with video:
                        video.settimeout(4)
                        for _ in range(30):
                            length, = struct.unpack(">I", exact(video, 4))
                            packet = exact(video, length)
                            self.assertTrue(packet.startswith(b"\0\0\0\1"))
                            images.extend(frame.to_ndarray(format="rgb24")
                                for frame in decoder.decode(av.Packet(packet)))
                        self.assertGreaterEqual(len(images), 29)
                        self.assertEqual(images[-1].shape, (720, 2560, 3))
                        # Two identical eye images, correct RGB and changing frames.
                        self.assertLess(np.abs(images[-1][:, :1280].astype(float)
                            - images[-1][:, 1280:].astype(float)).mean(), 2)
                        center = images[-1][360, 640]
                        self.assertGreater(int(center[0]), 190)
                        self.assertLess(int(center[2]), 40)
                        self.assertGreater(np.abs(images[-1].astype(float)
                            - images[5].astype(float)).mean(), 3)
                        publishing.clear()
                        publisher.join(timeout=2)
                        for _ in range(30):
                            length, = struct.unpack(">I", exact(video, 4))
                            decoded = decoder.decode(av.Packet(exact(video, length)))
                            if decoded:
                                last = decoded[-1].to_ndarray(format="rgb24")
                        # Stale source replaces the last image with a diagnostic.
                        self.assertLess(last[360, 640].mean(), 40)
                        ctrl.sendall(control_packet("CLOSE_CAMERA"))
                        self.assertEqual(video.recv(4), b"")
                    # Reopen in the same native control session.
                    ctrl.sendall(control_packet("OPEN_CAMERA", request))
                    video2, _ = listener.accept()
                    with video2:
                        video2.settimeout(4)
                        length, = struct.unpack(">I", exact(video2, 4))
                        self.assertGreater(len(exact(video2, length)), 100)
                    ctrl.sendall(control_packet("CLOSE_CAMERA"))
                    # Closing Listen may leave this control socket open. A new
                    # TCP connection must work without waiting for its FIN.
                    with socket.create_connection(("127.0.0.1", control_port), timeout=4) as ctrl2:
                        ctrl2.sendall(control_packet("OPEN_CAMERA", request))
                        video3, _ = listener.accept()
                        with video3:
                            video3.settimeout(4)
                            length, = struct.unpack(">I", exact(video3, 4))
                            decoded = av.CodecContext.create("h264", "r").decode(
                                av.Packet(exact(video3, length)))
                            self.assertEqual(decoded[0].width, 2560)
                            # Reconnect while the previous stream is ACTIVE,
                            # with a partial command left on its control socket.
                            ctrl2.sendall(b"\0\0")
                            with socket.create_connection(("127.0.0.1", control_port), timeout=4) as ctrl3:
                                ctrl3.sendall(control_packet("OPEN_CAMERA", request))
                                video4, _ = listener.accept()
                                with video4:
                                    video4.settimeout(4)
                                    length, = struct.unpack(">I", exact(video4, 4))
                                    decoded = av.CodecContext.create("h264", "r").decode(
                                        av.Packet(exact(video4, length)))
                                    self.assertEqual(decoded[0].height, 720)
                                    ctrl3.sendall(control_packet("CLOSE_CAMERA"))
            finally:
                publishing.clear()
                publisher.join(timeout=2)
                pub.close(linger=0)
                process.terminate()
                process.wait(timeout=6)
                logfile.seek(0)
                print(logfile.read())
                logfile.close()


if __name__ == "__main__":
    unittest.main()
