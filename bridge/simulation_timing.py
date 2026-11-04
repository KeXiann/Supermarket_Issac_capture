"""Wall-clock scheduling and measured physics throughput; no simulator imports."""
from collections import defaultdict, deque
import math
import time


class StartupPostureReadiness:
    """Neutral-pose readiness notice; not a balance or task-success certificate."""
    def __init__(self):
        self.stable_since = None

    def update(self, sim_time, tilt_rad, linear_velocity, angular_velocity):
        values = (sim_time, tilt_rad, *linear_velocity, *angular_velocity)
        stable = (all(math.isfinite(float(x)) for x in values)
                  and abs(tilt_rad) <= math.radians(15.)
                  and math.hypot(*linear_velocity[:2]) <= .25
                  and math.sqrt(sum(float(x)**2 for x in angular_velocity)) <= .75)
        if not stable:
            self.stable_since = None
            return False
        if self.stable_since is None or sim_time < self.stable_since:
            self.stable_since = sim_time
        return sim_time - self.stable_since >= 2.


class FrameSchedule:
    """Absolute deadlines prevent cumulative frame drift and catch-up bursts."""
    def __init__(self, fps):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        self.period = 1. / fps
        self.next_due = None

    def due(self, now):
        return self.next_due is None or now + 1e-9 >= self.next_due

    def captured(self, now):
        if self.next_due is None:
            self.next_due = now + self.period
        else:
            skipped = max(1, math.floor((now - self.next_due) / self.period) + 1)
            self.next_due += skipped * self.period


class PhysicsPacer:
    """Start pacing on actual physics, without startup/reset debt."""
    def __init__(self, dt, max_catchup_s=.02):
        self.dt = dt
        self.max_catchup_s = max_catchup_s
        self.anchor = None
        self.steps = 0
        self.rebases = 0
        self.discarded_debt_s = 0.

    def begin(self, now):
        if self.anchor is None:
            self.anchor = now

    def remaining(self, now):
        if self.anchor is None:
            return 0.
        self.steps += 1
        # A shader/JIT/OS stall must not accelerate the robot for seconds.
        # This changes only wall pacing: no skipped physics steps or fake ticks.
        lag = now - (self.anchor + self.steps * self.dt)
        if lag > self.max_catchup_s:
            debt = lag - self.max_catchup_s
            self.anchor += debt
            self.discarded_debt_s += debt
            self.rebases += 1
        return max(0., self.anchor + self.steps * self.dt - now)


class LoopPerformance:
    """Separate compute cost from throughput and startup/finalization."""
    def __init__(self, dt, profile=False):
        self.dt = dt
        self.profile = profile
        self.steps = 0
        self.first = self.last = None
        self.recent = deque(maxlen=2001)
        self.phases = defaultdict(lambda: deque(maxlen=20000))

    def begin(self):
        self.begin_wall = self.phase_wall = time.monotonic()
        self.pending = []

    def mark(self, name):
        if self.profile:
            now = time.monotonic()
            self.pending.append((name, now - self.phase_wall))
            self.phase_wall = now

    def finish(self, advanced):
        if not advanced:
            return
        now = time.monotonic()
        if self.first is None:
            self.first = self.begin_wall
            self.recent.append((self.first, 0))
        self.steps += 1
        self.last = now
        self.recent.append((now, self.steps))
        for name, seconds in self.pending:
            self.phases[name].append(seconds)

    def summary(self):
        span = max((self.last - self.first) if self.last is not None else 0., 1e-9)
        result = {'physics_dt_s': self.dt, 'physical_steps': self.steps,
                  'simulation_seconds': self.steps * self.dt, 'active_wall_seconds': span,
                  'real_time_factor': self.steps * self.dt / span,
                  'physics_steps_per_wall_second': self.steps / span,
                  'excludes': 'startup wait, inter-episode pause/reset, final encoding and disk flush'}
        if len(self.recent) >= 2:
            end, step = self.recent[-1]
            start, previous = next(((w, s) for w, s in self.recent if w >= end - 5.), self.recent[0])
            if end > start:
                result['recent_physics_steps_per_wall_second'] = (step - previous) / (end - start)
        if self.profile:
            def stats(values):
                data = sorted(values)
                return {'count': len(data), 'mean_ms': sum(data) / len(data) * 1000.,
                        'p95_ms': data[min(len(data)-1, int(.95 * len(data)))] * 1000.,
                        'max_ms': data[-1] * 1000.}
            result['phases'] = {name: stats(values) for name, values in self.phases.items() if values}
            result['phase_window_max_steps'] = 20000
        return result
