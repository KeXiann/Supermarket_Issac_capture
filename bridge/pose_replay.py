"""Time-based replay of recorded SMPL packets with a continuous stream clock."""
from collections import deque

import numpy as np


class PoseReplay:
    FIELDS = ("smpl_pose", "smpl_joints", "body_quat_w", "joint_pos")

    def __init__(self, path, start=0.):
        with np.load(path, allow_pickle=False) as archive:
            self.data = {key: archive[key].copy() for key in self.FIELDS}
            self.hands = {key: archive[key].copy() for key in ("left_hand_joints", "right_hand_joints") if key in archive}
            self.controllers = {key: archive[key].copy() for key in
                                ("left_trigger", "right_trigger", "left_grip", "right_grip") if key in archive}
            count = len(self.data["smpl_pose"])
            self.timeline = (archive["packet_time"]-archive["packet_time"][0]
                             if "packet_time" in archive else np.arange(count)/50.)
        shapes = ((count,21,3), (count,24,3), (count,4), (count,29))
        if count < 2 or any(self.data[k].shape != shape or not np.isfinite(self.data[k]).all()
                            for k, shape in zip(self.FIELDS, shapes)):
            raise ValueError("pose archive needs at least two finite SMPL frames with consistent shapes")
        for key, values in self.hands.items():
            if values.shape != (count, 7) or not np.isfinite(values).all():
                raise ValueError(f"invalid recorded hand reference: {key}")
        for key, values in self.controllers.items():
            if values.shape != (count, 1) or not np.isfinite(values).all():
                raise ValueError(f"invalid recorded controller reference: {key}")
        if (self.timeline.shape != (count,) or not np.isfinite(self.timeline).all()
                or np.any(np.diff(self.timeline) <= 0)):
            raise ValueError("pose timestamps must be finite and strictly increasing")
        self.duration = float(self.timeline[-1])
        if not np.isfinite(start) or not 0. <= start < self.duration:
            raise ValueError(f"pose-start must be within 0..{self.duration:.3f}s")
        self.start = start
        self.buffer = {key: deque(maxlen=5) for key in self.FIELDS}
        self.sent = 0
        self.last_source_frame = None

    def run_seconds(self, requested, warmup, tail):
        if not np.isfinite([requested, warmup, tail]).all() or min(requested, warmup, tail) < 0:
            raise ValueError("replay timing arguments must be finite and nonnegative")
        return requested or warmup+self.duration-self.start+tail

    def payload(self, elapsed, monotonic):
        frame = int(np.clip(np.searchsorted(self.timeline, self.start+elapsed, side="right")-1,
                            0, len(self.timeline)-1))
        for key in self.FIELDS:
            value = self.data[key][frame]
            if not self.buffer[key]:
                self.buffer[key].extend([value]*4)
            self.buffer[key].append(value)
        result = {key: np.stack(self.buffer[key]) for key in self.FIELDS}
        result.update({key: values[frame].astype(np.float32) for key, values in self.hands.items()})
        result.update({key: values[frame].astype(np.float32) for key, values in self.controllers.items()})
        # Advance the stream even at the end of the archive. SONIC needs future
        # frames to consume its final reference window; repeating the original
        # source index would leave the last recorded motion stuck behind it.
        result.update(joint_vel=np.zeros((5,29), dtype=np.float32),
                      frame_index=np.arange(self.sent-4, self.sent+1, dtype=np.int64),
                      timestamp_monotonic=np.array([monotonic], dtype=np.float64))
        self.sent += 1
        self.last_source_frame = frame
        return result
