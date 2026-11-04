"""G1 simulated ego camera and camera-streaming adapters.

The Isaac bridge owns rendering, so this module deliberately keeps all Isaac
imports inside the setup/capture methods.  The resulting RGB frames are
compatible with GR00T's ``ego_view`` camera protocol. pico_video_server adapts
this stream to XRoboToolkit's native Listen receiver; RTP is optional.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


EGO_CAMERA_PATH = "/World/TeleopRender/EgoCamera"
EGO_CAMERA_PROFILE = "g1_balanced_grasp_ego_v4"
EGO_CAMERA_MOUNT_M = (0.10, 0.0, 0.35)
EGO_CAMERA_PITCH_DOWN_DEG = 45.0
EGO_CAMERA_FOCAL_LENGTH_MM = 14.0


def setup_ego_camera(stage, parent_path: str, *, camera_path: str = EGO_CAMERA_PATH,
                     width: int = 640, height: int = 480,
                     mount_position_m=EGO_CAMERA_MOUNT_M,
                     pitch_down_deg: float = EGO_CAMERA_PITCH_DOWN_DEG,
                     focal_length_mm: float = EGO_CAMERA_FOCAL_LENGTH_MM,
                     intrinsics=None, extrinsics=None, clipping_range_m=(0.02, 100.0)) -> dict[str, Any]:
    """Create a torso camera; explicit mount estimates follow the full rigid pose.

    Legacy calls without extrinsics retain their world-up operator view so old
    captures can still be reproduced. Neither mode certifies physical accuracy.
    """
    from pxr import Gf, Sdf, UsdGeom
    from camera_intrinsics import usd_projection_parameters, centered_render_plan

    if extrinsics:
        from camera_extrinsics import validate_extrinsics
        local_cv=validate_extrinsics(extrinsics)
        if parent_path.rsplit('/',1)[-1]!='torso_link':
            raise ValueError('Estimated torso camera cannot be attached to a different parent frame')
        mount_position_m=local_cv[:3,3]
        pitch_down_deg=math.degrees(math.atan2(-local_cv[2,2],math.hypot(local_cv[0,2],local_cv[1,2])))
    mount = tuple(float(v) for v in mount_position_m)
    if (len(mount) != 3 or not all(math.isfinite(v) for v in mount)
            or not math.isfinite(pitch_down_deg) or not 0 <= pitch_down_deg < 85
            or not math.isfinite(focal_length_mm) or focal_length_mm <= 0
            or width < 16 or height < 16):
        raise ValueError("Camera requires finite xyz mount, pitch in [0,85), positive focal length and dimensions >=16")
    render_plan = centered_render_plan(intrinsics,width,height) if intrinsics else None
    if intrinsics:
        rw,rh = render_plan['render_width'],render_plan['render_height']
        film = {**intrinsics,'width':rw,'height':rh,
                'K_px': [[intrinsics['K_px'][0][0],0.,rw/2], [0.,intrinsics['K_px'][1][1],rh/2], [0.,0.,1.]]}
        projection = usd_projection_parameters(film,rw,rh,focal_length_mm)
    else:
        projection = {'horizontal_aperture_mm':36., 'vertical_aperture_mm':36.*height/width,
                      'horizontal_aperture_offset_mm':0., 'vertical_aperture_offset_mm':0.}
    direction = tuple(local_cv[:3,2]) if extrinsics else (1.0,0.0,-math.tan(math.radians(pitch_down_deg)))

    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        raise RuntimeError(f"Cannot attach ego camera: missing body prim {parent_path}")
    stage.SetEditTarget(stage.GetSessionLayer())
    # Keep the camera at a stable top-level path.  The bridge updates its
    # world transform from the articulated body each render cycle; this is
    # more reliable than relying on a Camera prim nested below a PhysX body.
    camera_path = EGO_CAMERA_PATH if camera_path == EGO_CAMERA_PATH else camera_path
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateHorizontalApertureAttr(projection['horizontal_aperture_mm'])
    camera.CreateVerticalApertureAttr(projection['vertical_aperture_mm'])
    camera.CreateHorizontalApertureOffsetAttr(projection['horizontal_aperture_offset_mm'])
    camera.CreateVerticalApertureOffsetAttr(projection['vertical_aperture_offset_mm'])
    # Use a centered, undistorted renderer and explicitly remap the guard-band
    # image. The actual installed RTX renderer ignored off-axis lens metadata.
    prim = camera.GetPrim()
    prim.RemoveAppliedSchema('OmniLensDistortionOpenCvPinholeAPI')
    for name in list(prim.GetPropertyNames()):
        if name.startswith('omni:lensdistortion:'):
            prim.RemoveProperty(name)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*clipping_range_m))
    UsdGeom.Xformable(camera).ClearXformOpOrder()
    # Keep a readable tag in the session layer for metadata/auditing.  Custom
    # USD metadata keys are rejected unless registered, so use a namespaced
    # string attribute instead.
    camera.GetPrim().CreateAttribute(
        "teleop:egoCameraParent", Sdf.ValueTypeNames.String
    ).Set(parent_path)
    info = {
        "profile": EGO_CAMERA_PROFILE,
        "view_kind": ("rectified_intrinsics_photo_cad_estimated_rigid_mount" if extrinsics and intrinsics else
                      "rectified_real_camera_intrinsics_existing_virtual_mount" if intrinsics else "virtual_balanced_grasp_view"),
        "physical_camera_calibration": False,
        "intrinsics_match_real_rectified_output": bool(intrinsics),
        "extrinsics_calibrated": False,
        "extrinsics": extrinsics,
        "T_parent_camera_cv": extrinsics['T_parent_camera_cv'] if extrinsics else None,
        "mount_rotation_semantics": "rigid_parent" if extrinsics else "legacy_world_up_stabilized",
        "intrinsics": intrinsics,
        "render_plan": render_plan,
        "clipping_range_m": list(clipping_range_m),
        "camera_path": camera_path,
        "parent_path": parent_path,
        "mount_position_m": mount,
        "local_view_direction": direction,
        "pitch_down_deg": float(pitch_down_deg),
        "width": int(width),
        "height": int(height),
        "focal_length_mm": float(focal_length_mm),
        "focal_length_mm_kind": "nominal USD representation; use intrinsics.K_px for calibrated output" if intrinsics else "legacy virtual lens setting",
        **projection,
        "session_layer_only": True,
    }
    update_ego_camera_transform(stage, info)
    return info


def _transform_point(matrix, point):
    from pxr import Gf

    # Gf.Matrix4d stores USD transforms with translation in the final row and
    # applies points through its row-vector Transform helper.  Multiplying a
    # column Vec4d here silently dropped the torso world translation.
    return matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))


def update_ego_camera_transform(stage, info: dict[str, Any], *,
                                world_pos=None, world_quat_wxyz=None) -> None:
    """Copy the current torso world pose to the top-level ego camera.

    IsaacLab's articulation state is the authoritative pose while the robot is
    moving.  The optional numerical pose avoids depending on whether the USD
    body prim has received the latest PhysX/Fabric transform yet.
    """
    from pxr import Gf, Usd, UsdGeom

    body = stage.GetPrimAtPath(info["parent_path"])
    camera = stage.GetPrimAtPath(info["camera_path"])
    if not body or not body.IsValid() or not camera or not camera.IsValid():
        raise RuntimeError("Ego camera/body prim disappeared while rendering")
    if world_pos is not None and world_quat_wxyz is not None:
        rotation = Gf.Rotation(Gf.Quatd(
            float(world_quat_wxyz[0]), float(world_quat_wxyz[1]),
            float(world_quat_wxyz[2]), float(world_quat_wxyz[3])))
        body_world = Gf.Matrix4d().SetRotate(rotation)
        # SetTranslate resets the rotation to identity in Gf.Matrix4d. Preserve
        # the torso heading, especially the living-room robot's 90-degree yaw.
        body_world.SetTranslateOnly(Gf.Vec3d(
            float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))
    else:
        body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if info.get('extrinsics'):
        from camera_extrinsics import validate_extrinsics
        transform=validate_extrinsics(info['extrinsics']).copy()
        # OpenCV (+X right,+Y down,+Z forward) to USD camera
        # (+X right,+Y up,-Z forward); convert column to Gf row convention.
        transform[:,1:3]*=-1
        local_usd=Gf.Matrix4d(tuple(tuple(float(transform[j,i]) for j in range(4)) for i in range(4)))
        camera_world=local_usd*body_world
    else:
        # Preserve the historical world-up renderer when replaying old captures.
        mount = info["mount_position_m"]
        direction = info["local_view_direction"]
        eye = _transform_point(body_world, mount)
        forward_end = _transform_point(body_world, tuple(mount[i] + direction[i] for i in range(3)))
        camera_world = Gf.Matrix4d().SetLookAt(
            eye, forward_end, Gf.Vec3d(0.0, 0.0, 1.0)
        ).GetInverse()
    camera_xform = UsdGeom.Xformable(camera)
    ops = camera_xform.GetOrderedXformOps()
    if not ops:
        ops = [camera_xform.AddTransformOp()]
    ops[0].Set(camera_world)


class EgoCameraCapture:
    """Replicator reader plus MP4, ZMQ and optional RTP outputs."""

    def __init__(self, camera_path: str, *, width: int, height: int, fps: int,
                 record_path: str = "", zmq_port: int = 0,
                 rtp_host: str = "", rtp_port: int = 0, camera_info=None) -> None:
        import cv2
        import omni.replicator.core as rep

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._cv2 = cv2
        self._rep = rep
        plan = camera_info.get('render_plan') if camera_info else None
        from camera_intrinsics import RectifiedOutputMapper
        self._mapper = RectifiedOutputMapper(plan) if plan else None
        self.render_width = plan['render_width'] if plan else self.width
        self.render_height = plan['render_height'] if plan else self.height
        self._product = rep.create.render_product(camera_path, (self.render_width, self.render_height))
        self._reader = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._reader.attach(self._product)
        self._camera_info = camera_info
        self._projection_reader = None
        if plan and camera_info.get('capture_contract'):
            self._projection_reader = rep.AnnotatorRegistry.get_annotator('camera_params')
            self._projection_reader.attach(self._product)
        self._last_capture = 0.0
        from simulation_timing import FrameSchedule
        self._capture_schedule = FrameSchedule(self.fps)
        self._frame_count = 0
        self._started_wall = time.time()
        self._writer = None
        self._timestamps = None
        self._zmq_context = None
        self._zmq_socket = None
        self._rtp = None
        self._record_path = str(record_path) if record_path else ""
        self._record_closed = False
        if self._record_path:
            self.start_recording(self._record_path)
        if int(zmq_port) > 0:
            import msgpack
            import zmq

            self._msgpack = msgpack
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.PUB)
            self._zmq_socket.setsockopt(zmq.SNDHWM, 3)
            self._zmq_socket.setsockopt(zmq.LINGER, 0)
            self._zmq_socket.bind(f"tcp://*:{int(zmq_port)}")
            print(f"[EGO] ego_view ZMQ publisher: tcp://*:{int(zmq_port)}", flush=True)
        if int(rtp_port) > 0:
            if not rtp_host:
                raise ValueError("rtp_port requires a non-empty rtp_host")
            self._rtp = self._start_rtp(rtp_host, int(rtp_port))
            print(f"[EGO] ego_view RTP sender: {rtp_host}:{int(rtp_port)}", flush=True)
        from frame_output import FrameOutput
        self._output = FrameOutput(self._write_frame)

    def start_recording(self, record_path):
        """Rotate only file outputs; preserve the render product and PICO stream."""
        if hasattr(self, '_output'):
            self._output.drain()
        if self._writer is not None or self._timestamps is not None:
            raise RuntimeError('finish the previous ego recording before rotating')
        path = Path(record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if any(p.exists() for p in (path, path.with_suffix('.timestamps.csv'), path.with_suffix('.recording.json'))):
            raise FileExistsError(f'Refusing to overwrite ego recording: {path}')
        self._record_path = str(path)
        self._record_closed = False
        self._frame_count = 0
        self._started_wall = time.time()
        self._last_capture = 0.
        from simulation_timing import FrameSchedule
        self._capture_schedule = FrameSchedule(self.fps)
        self._writer = self._cv2.VideoWriter(str(path), self._cv2.VideoWriter_fourcc(*'mp4v'),
                                            self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f'Cannot open ego-view video writer: {path}')
        self._timestamps = path.with_suffix('.timestamps.csv').open('x', newline='', encoding='utf-8')
        self._timestamps_writer = csv.writer(self._timestamps)
        self._timestamps_writer.writerow(['frame', 'sim_time_s', 'wall_time_s', 'unix_time_s', 'host_monotonic_s',
                                         'render_complete_monotonic_s', 'source_physics_step'])

    def _start_rtp(self, host: str, port: int):
        # camera_viz expects H.264/RTP.  FFmpeg is used as a small, portable
        # bridge because Isaac's Python environment does not ship PyGObject.
        command = [
            "ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-tune", "zerolatency",
            "-profile:v", "baseline", "-g", str(max(self.fps, 1)), "-bf", "0",
            "-f", "rtp", f"rtp://{host}:{port}?pkt_size=1400",
        ]
        try:
            return subprocess.Popen(command, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    bufsize=0)
        except OSError as exc:
            raise RuntimeError("RTP ego-view output requires ffmpeg on PATH") from exc

    def _jpeg_base64(self, rgb):
        ok, encoded = self._cv2.imencode(
            ".jpg", rgb, [int(self._cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode ego_view frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def map_rendered_image(self, image, *, depth=False):
        return self._mapper.apply(image, depth=depth) if self._mapper else image

    def read_rgb(self):
        rgb = self._reader.get_data()
        if rgb is None or not rgb.size:
            return None
        rgb = self.map_rendered_image(rgb[..., :3])
        if rgb.shape[:2] != (self.height, self.width):
            raise RuntimeError(f"Unexpected ego camera frame shape: {rgb.shape}")
        return rgb

    def capture_due(self, now=None):
        """Check before rendering, rather than discarding an expensive RTX frame."""
        now = time.monotonic() if now is None else now
        return self._last_capture == 0. or self._capture_schedule.due(now)

    def maybe_capture(self, sim_time: float, *, force=False, sample_monotonic_ns=None, sample_unix_ns=None,
                      source_physics_step=-1) -> bool:
        self._output.check()
        now = time.monotonic()
        if not force and not self.capture_due(now):
            return False
        if self._last_capture == 0.:
            from simulation_timing import FrameSchedule
            self._capture_schedule = FrameSchedule(self.fps)
        self._last_capture = now
        self._capture_schedule.captured(now)
        # An empty annotator during RTX warmup still consumes a render slot;
        # retry at the next deadline rather than rendering at every 200 Hz step.
        rgb = self.read_rgb()
        if rgb is None:
            return False
        # Copy the annotator buffer before the next RTX update can reuse it.
        # Capture clocks stay on this thread; encoding latency cannot retime a frame.
        unix_time = time.time() if sample_unix_ns is None else sample_unix_ns / 1e9
        sample_time = now if sample_monotonic_ns is None else sample_monotonic_ns / 1e9
        self._output.submit((rgb.copy(), float(sim_time), unix_time, sample_time,
                             time.time() - self._started_wall, self._frame_count, now, source_physics_step))
        self._frame_count += 1
        return True

    def _write_frame(self, frame):
        rgb, sim_time, unix_time, now, wall_time, frame_index, render_complete, source_step = frame
        if self._writer is not None:
            self._writer.write(self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))
            self._timestamps_writer.writerow([
                frame_index, f"{float(sim_time):.6f}",
                f"{wall_time:.6f}", f"{unix_time:.6f}",
                f"{now:.9f}", f"{render_complete:.9f}", int(source_step),
            ])
            self._timestamps.flush()
        if self._zmq_socket is not None:
            payload = {
                "timestamps": {"ego_view": unix_time},
                "images": {"ego_view": self._jpeg_base64(rgb)},
                "jpeg_encoder_input": "rgb",
            }
            try:
                self._zmq_socket.send(self._msgpack.packb(payload, use_bin_type=True),
                                      flags=1)
            except Exception:
                # A disconnected SUB is normal; never stall physics for a viewer.
                pass
        if self._rtp is not None and self._rtp.stdin is not None:
            try:
                self._rtp.stdin.write(rgb.tobytes())
            except (BrokenPipeError, OSError):
                self._rtp = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def verify_projection(self):
        if self._projection_reader is None:
            raise RuntimeError('Projection verification requires an upstream capture contract')
        from camera_intrinsics import verify_native_projection
        result = verify_native_projection(self._projection_reader.get_data(), self._camera_info)
        self._projection_reader.detach(self._product)
        self._projection_reader = None
        return result

    def finish_recording(self) -> None:
        """Durably finish MP4 and timestamps before any native Kit cleanup."""
        if self._record_closed:
            return
        output_error = None
        try:
            self._output.drain()
        except RuntimeError as exc:
            output_error = exc
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._timestamps is not None:
            self._timestamps.flush()
            os.fsync(self._timestamps.fileno())
            self._timestamps.close()
            self._timestamps = None
        if self._record_path:
            path = Path(self._record_path)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            status = path.with_suffix(".recording.json")
            temporary = status.with_suffix(".tmp")
            with temporary.open("x") as stream:
                json.dump({"closed": output_error is None, "frames": self._frame_count,
                           "output_error": str(output_error) if output_error else None,
                           "output_worker": "ordered bounded FIFO; drained before MP4/timestamp close",
                           "nominal_fps": self.fps, "width": self.width, "height": self.height,
                           "clock": "host_monotonic_s: same host boot as Bridge/Manager; sim_time_s: physics clock",
                           "playback": "raw MP4 uses nominal FPS; use timestamped replay for original timing"}, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(status)
        self._record_closed = True
        if output_error:
            raise output_error

    def close(self) -> None:
        output_error = None
        try:
            self.finish_recording()
        except RuntimeError as exc:
            output_error = exc
        try:
            self._output.close()
        except RuntimeError as exc:
            output_error = exc
        if self._zmq_socket is not None:
            self._zmq_socket.close()
            self._zmq_socket = None
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None
        if self._rtp is not None:
            if self._rtp.stdin is not None:
                self._rtp.stdin.close()
            self._rtp.terminate()
            try:
                self._rtp.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._rtp.kill()
            self._rtp = None
        if self._reader is not None and self._product is not None:
            if self._projection_reader is not None:
                self._projection_reader.detach(self._product)
                self._projection_reader = None
            self._reader.detach(self._product)
            self._product.destroy()
            self._reader = None
            self._product = None
        if output_error:
            raise output_error
