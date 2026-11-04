#!/usr/bin/env python3
"""Isaac ego_view -> XRoboToolkit native Remote Vision / Listen.

Implements the upstream XR-Robotics Orin sender's wire protocol, without a
physical ZED camera: BE32-framed control messages containing LE32 command and
payload lengths; CAFE/v1 camera request; BE32-framed Annex-B H.264 access units
on a reverse connection to the headset. Existing pose tracking stays separate.
See XR-Robotics/XRoboToolkit-Orin-Video-Sender/main_zed_tcp.cpp and
XR-Robotics/XRoboToolkit-Unity-Client/Assets/Scripts/Network/CameraRequestSerializer.cs.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import struct
import threading
import time

import av
import msgpack
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import zmq

SERVER_CODE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def send_video_packet(sock, payload, stop, *, stall_seconds=10., on_stall=None):
    """Finish one length-prefixed access unit without duplicating partial bytes.

    A short socket timeout is a cancellation check, not a stream failure.
    sendall() cannot safely be retried after timeout: it hides how many bytes
    already reached the receiver. Retain the offset until this frame finishes.
    A real stalled connection is closed by the caller and gets a fresh encoder.
    """
    wire = memoryview(struct.pack(">I", len(payload)) + payload)
    offset = 0
    started = last_progress = time.monotonic()
    notified = False
    while offset < len(wire):
        if stop.is_set():
            raise InterruptedError("video session cancelled")
        now = time.monotonic()
        if now - last_progress >= stall_seconds or now - started >= 2 * stall_seconds:
            raise TimeoutError(f"video transport stalled at {offset}/{len(wire)} bytes")
        try:
            count = sock.send(wire[offset:offset + 65536])
        except (socket.timeout, BlockingIOError):
            if not notified and on_stall is not None:
                on_stall()
                notified = True
            continue
        if count == 0:
            raise ConnectionError("headset closed the video socket")
        offset += count
        last_progress = time.monotonic()
    return time.monotonic() - started


def log(message):
    print(time.strftime("[%H:%M:%S] [PICO VIDEO] ") + message, flush=True)


def read_exact(sock, count, stop=None):
    data = bytearray()
    while len(data) < count:
        if stop is not None and stop.is_set():
            raise EOFError("stopped")
        try:
            chunk = sock.recv(count - len(data))
        except socket.timeout:
            continue
        if not chunk:
            raise EOFError("peer disconnected")
        data.extend(chunk)
    return bytes(data)


def read_command(sock, stop=None):
    size, = struct.unpack(">I", read_exact(sock, 4, stop))
    if not 8 <= size <= 65536:
        raise ValueError(f"invalid control frame size {size}")
    data = read_exact(sock, size, stop)
    cmd_size, = struct.unpack_from("<I", data)
    if not 1 <= cmd_size <= 128 or 8 + cmd_size > len(data):
        raise ValueError("invalid command length")
    command = data[4:4 + cmd_size].rstrip(b"\0").decode("utf-8")
    payload_size, = struct.unpack_from("<I", data, 4 + cmd_size)
    if 8 + cmd_size + payload_size != len(data):
        raise ValueError("control payload length mismatch")
    return command, data[8 + cmd_size:]


def pack_command(command, payload):
    cmd = command.encode("utf-8")
    body = struct.pack("<I", len(cmd)) + cmd + struct.pack("<I", len(payload)) + payload
    return struct.pack(">I", len(body)) + body


@dataclass
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    hevc: int
    render_mode: int
    port: int
    camera: str
    ip: str

    @classmethod
    def parse(cls, data):
        if len(data) < 33 or data[:3] != b"\xca\xfe\x01":
            raise ValueError("expected CAFE camera request version 1")
        values = struct.unpack_from("<7i", data, 3)
        strings = []
        offset = 31
        for _ in range(2):
            if offset >= len(data):
                raise ValueError("missing camera request string")
            size = data[offset]
            offset += 1
            if offset + size > len(data):
                raise ValueError("truncated camera request string")
            strings.append(data[offset:offset + size].decode("utf-8"))
            offset += size
        req = cls(*values, *strings)
        if not (64 <= req.width <= 4096 and 64 <= req.height <= 2160
                and req.width % 4 == 0 and req.height % 2 == 0
                and 1 <= req.fps <= 120 and 1 <= req.port <= 65535):
            raise ValueError(f"unsupported dimensions, fps or callback port: {req}")
        if req.hevc:
            raise ValueError("H.264 is required; choose ZEDMINI in Remote Vision")
        if req.camera not in ("ZED", "VR"):
            raise ValueError("choose ZEDMINI in Remote Vision (flat SBS image)")
        return req


class LatestImage:
    """Continuously drain the PUB stream; never queue old frames for the HMD."""
    def __init__(self, endpoint, stop):
        self.endpoint = endpoint
        self.stop = stop
        self.lock = threading.Lock()
        self.latest = None
        self.count = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        with zmq.Context() as context, context.socket(zmq.SUB) as sub:
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.setsockopt(zmq.CONFLATE, 1)
            sub.setsockopt(zmq.LINGER, 0)
            sub.connect(self.endpoint)
            last_error = 0.0
            while not self.stop.is_set():
                if not sub.poll(200):
                    continue
                try:
                    payload = msgpack.unpackb(sub.recv(), raw=False)
                    jpeg = base64.b64decode(payload["images"]["ego_view"])
                    stamp = float(payload["timestamps"]["ego_view"])
                    # GR00T ImageUtils passes RGB arrays directly to cv2.imencode.
                    # Pillow interprets that JPEG as BGR, so undo the swap here.
                    # The sender can explicitly declare a conventional JPEG.
                    order = payload.get("jpeg_encoder_input", "rgb")
                    with self.lock:
                        self.latest = (jpeg, stamp, order)
                        self.count += 1
                except Exception as exc:
                    if time.monotonic() - last_error > 5:
                        log(f"source decode error: {exc}")
                        last_error = time.monotonic()

    def snapshot(self):
        with self.lock:
            return self.latest, self.count


class VideoSession:
    def __init__(self, req, peer, source, status, max_fps=30, stale_seconds=2.0,
                 max_bitrate=2_500_000, send_stall_seconds=10., send_buffer_bytes=131072):
        self.req, self.peer, self.source, self.status = req, peer, source, status
        self.fps = min(req.fps, max_fps)
        self.stale_seconds = stale_seconds
        self.max_bitrate = max_bitrate
        self.send_stall_seconds = send_stall_seconds
        self.send_buffer_bytes = send_buffer_bytes
        self.frames = 0
        self.reconnects = 0
        self.backpressure_events = 0
        self.effective_fps = self.fps
        self.last_error = ""
        self.stop_event = threading.Event()
        self.sock = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=4)

    def status_write(self, **values):
        if self.status:
            latest, count = self.source.snapshot()
            age = time.time() - latest[1] if latest else None
            source_live = age is not None and -2 <= age <= self.stale_seconds
            target = Path(self.status)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps({"updated_unix_s": time.time(),
                "server_pid": os.getpid(), "server_code_hash": SERVER_CODE_HASH,
                "headset_ip": self.peer, "source_frames": count,
                "source_age_s": round(age, 3) if age is not None else None,
                "source_live": source_live, "sent_frames": self.frames,
                "reconnect_count": self.reconnects, "backpressure_events": self.backpressure_events,
                "effective_fps": self.effective_fps, "last_error": self.last_error,
                **values}, indent=2) + "\n")
            temporary.replace(target)

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    # Reply to the requesting peer; avoid stale cached local IPs.
                    self.sock = socket.create_connection((self.peer, self.req.port), timeout=1)
                    self.sock.settimeout(.2)
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer_bytes)
                    self.stream_connected()
                except (OSError, EOFError) as exc:
                    if self.stop_event.is_set():
                        break
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.reconnects += 1
                    log(f"VIDEO RECONNECT {self.reconnects}: {self.last_error}; source subscription retained")
                    self.status_write(connected=False, transport_state="reconnecting")
                finally:
                    sock, self.sock = self.sock, None
                    if sock is not None:
                        sock.close()
                # Each callback reconnect starts a fresh H264 IDR/header stream.
                self.stop_event.wait(min(3., .5 + self.reconnects * .25))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log(f"VIDEO STOPPED: {exc}")
        finally:
            self.status_write(connected=False, transport_state="stopped" if self.stop_event.is_set() else "error")

    def stream_connected(self):
        codec = av.CodecContext.create("libx264", "w")
        codec.width, codec.height = self.req.width, self.req.height
        codec.pix_fmt = "yuv420p"
        codec.time_base = Fraction(1, self.fps)
        codec.framerate = Fraction(self.fps, 1)
        codec.bit_rate = max(500_000, min(self.req.bitrate, self.max_bitrate))
        codec.options = {"preset": "ultrafast", "tune": "zerolatency",
            "profile": "baseline", "x264-params":
            "repeat-headers=1:aud=1:keyint=15:min-keyint=15:scenecut=0:bframes=0:threads=2"}
        codec.open()
        log(f"VIDEO CONNECTED {self.peer}:{self.req.port}; H264 "
            f"{codec.width}x{codec.height}@{self.fps} bitrate={codec.bit_rate}; duplicated mono SBS")
        start = time.monotonic()
        next_frame, last_report, last_status = start, start - 5, start - 1
        last_count, eye, previous_stale = -1, None, None
        connection_frames = 0
        stable_since = start
        while not self.stop_event.is_set():
            latest, count = self.source.snapshot()
            age = time.time() - latest[1] if latest else float("inf")
            stale = latest is None or age > self.stale_seconds or age < -2
            if stale:
                eye = Image.new("RGB", (codec.width // 2, codec.height), (18, 20, 25))
                draw = ImageDraw.Draw(eye)
                draw.text((30, 40), "NO LIVE ISAAC FRAMES", fill=(255, 100, 70), font_size=28)
                draw.text((30, 90), "Start / restart bridge-livingroom", fill="white", font_size=22)
                draw.text((30, 130), time.strftime("%H:%M:%S"), fill="white", font_size=22)
            elif count != last_count or previous_stale:
                decoded = Image.open(io.BytesIO(latest[0])).convert("RGB")
                if latest[2] == "rgb":
                    decoded = Image.fromarray(np.asarray(decoded)[..., ::-1].copy())
                eye = ImageOps.pad(decoded, (codec.width // 2, codec.height), color="black")
            if stale != previous_stale:
                log("SOURCE STALE: diagnostic image displayed" if stale else "SOURCE LIVE: Isaac ego_view received")
            previous_stale, last_count = stale, count
            stereo = Image.new("RGB", (codec.width, codec.height))
            stereo.paste(eye, (0, 0))
            stereo.paste(eye, (codec.width // 2, 0))
            frame = av.VideoFrame.from_image(stereo)
            frame.pts = connection_frames
            send_seconds = 0.
            def backpressure():
                self.backpressure_events += 1
                log("VIDEO BACKPRESSURE: preserving the partial frame while waiting for headset")
                self.status_write(connected=True, transport_state="backpressure")
            for packet in codec.encode(frame):
                send_seconds += send_video_packet(self.sock, bytes(packet), self.stop_event,
                    stall_seconds=self.send_stall_seconds, on_stall=backpressure)
            self.frames += 1
            connection_frames += 1
            now = time.monotonic()
            if send_seconds > max(.1, 3 / self.effective_fps):
                self.effective_fps = max(min(10, self.fps), int(self.effective_fps * .75))
                stable_since = now
            elif now - stable_since > 5 and self.effective_fps < self.fps:
                self.effective_fps = min(self.fps, self.effective_fps + 2)
                stable_since = now
            if now - last_status >= 1:
                self.status_write(connected=True, transport_state="streaming")
                last_status = now
            if now - last_report >= 5:
                log(f"sent={self.frames} connection_fps={connection_frames / max(now-start, .001):.1f} "
                    f"pace={self.effective_fps} source_frames={count} source_age={age:.2f}s live={not stale}")
                last_report = now
            next_frame += 1 / self.effective_fps
            if now - next_frame > 1 / self.effective_fps:
                next_frame = now
            self.stop_event.wait(max(0, next_frame - time.monotonic()))


class ControlConnection:
    """Own one control socket; the accept loop must never wait for its reads.

    PICO can retain the old TCP socket after CLOSE_CAMERA, or reconnect with a
    different Wi-Fi address without FIN on the previous socket. A new control
    connection supersedes that old owner, including its reverse video socket.
    """
    def __init__(self, conn, address, source, args):
        self.conn, self.address, self.source, self.args = conn, address, source, args
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError("previous video control owner did not stop")

    def run(self):
        session = None
        audio_request_id = None
        log(f"CONTROL CONNECTED {self.address[0]}:{self.address[1]}")
        try:
            with self.conn:
                self.conn.settimeout(.5)
                while not self.stop.is_set():
                    command, payload = read_command(self.conn, self.stop)
                    if command == "OPEN_CAMERA":
                        req = CameraRequest.parse(payload)
                        log(f"OPEN_CAMERA {req}")
                        if session is not None:
                            session.close()
                        # New clients optionally negotiate projection alongside
                        # audio. Older clients need no ACK and remain compatible.
                        if audio_request_id:
                            ack = {"schema": "g1_wuji_audio_ports_v2",
                                "audio_request_id": audio_request_id,
                                "audio_stream_port": 0, "microphone_upload_port": 0,
                                "video_projection": "flat", "video_stereo_layout": "side_by_side"}
                            self.conn.sendall(pack_command("AUDIO_CONFIG", json.dumps(ack).encode()))
                        session = VideoSession(req, self.address[0], self.source,
                            self.args.status, self.args.max_fps, self.args.stale_seconds,
                            self.args.max_bitrate, self.args.send_stall_seconds, self.args.send_buffer_bytes)
                    elif command == "CLOSE_CAMERA":
                        if session is not None:
                            session.close()
                            session = None
                        log("CLOSE_CAMERA")
                    elif command == "AUDIO_SESSION":
                        audio_request_id = payload.decode("ascii")
                    else:
                        log(f"ignored control command {command}")
        except (EOFError, OSError, ValueError, struct.error) as exc:
            log(f"CONTROL CLOSED {self.address[0]}:{self.address[1]}: {exc}")
        finally:
            if session is not None:
                session.close()


def serve(args):
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    source = LatestImage(args.source, stop)
    active = None
    try:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(4)
            server.settimeout(.5)
            log(f"READY control={args.host}:{args.port} source={args.source} code={SERVER_CODE_HASH}")
            if args.status:
                target = Path(args.status)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({"updated_unix_s": time.time(),
                    "server_pid": os.getpid(), "server_code_hash": SERVER_CODE_HASH,
                    "connected": False, "transport_state": "waiting_for_listen",
                    "source_live": False, "sent_frames": 0,
                    "reconnect_count": 0, "backpressure_events": 0}, indent=2) + "\n")
            log("PICO: Remote Vision -> ZEDMINI -> Listen -> workstation IP (no browser)")
            while not stop.is_set():
                try:
                    conn, address = server.accept()
                except socket.timeout:
                    continue
                if active is not None:
                    log(f"CONTROL REPLACE {active.address[0]}:{active.address[1]} -> "
                        f"{address[0]}:{address[1]}")
                    active.close()
                active = ControlConnection(conn, address, source, args)
    finally:
        if active is not None:
            active.close()
        stop.set()
        source.thread.join(timeout=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tcp://127.0.0.1:5555")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=13579)
    parser.add_argument("--max-fps", type=int, default=30)
    parser.add_argument("--max-bitrate", type=int, default=2_500_000)
    parser.add_argument("--send-stall-seconds", type=float, default=10.)
    parser.add_argument("--send-buffer-bytes", type=int, default=131072)
    parser.add_argument("--stale-seconds", type=float, default=2)
    parser.add_argument("--status", default="")
    args = parser.parse_args()
    if not 1 <= args.max_fps <= 120 or not 500_000 <= args.max_bitrate <= 12_000_000:
        parser.error("max-fps must be 1..120 and max-bitrate must be 500000..12000000")
    if args.send_stall_seconds <= 0 or args.send_buffer_bytes < 4096:
        parser.error("send-stall-seconds must be positive and send-buffer-bytes must be >=4096")
    serve(args)
