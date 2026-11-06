"""Object-relative scan goals, base placement, and measured arrival checks.

This module does not command a robot. Replace RandomViewPlanner.next_view with
an external next-best-view program; call it only after the previous capture.
All poses use metres and scalar-first (wxyz) quaternions.
"""

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread

import numpy as np
from scipy.spatial.transform import Rotation


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])


def quaternion(rot):
    return rot.as_quat()[[3, 0, 1, 2]]


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Pose:
    position: np.ndarray
    quaternion: np.ndarray

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).copy()
        self.quaternion = np.asarray(self.quaternion, dtype=float).copy()
        if self.position.shape != (3,) or self.quaternion.shape != (4,):
            raise ValueError("A pose requires position[3] and quaternion_wxyz[4]")
        if not np.isfinite(self.position).all() or not np.isfinite(self.quaternion).all():
            raise ValueError("Pose contains non-finite values")
        norm = np.linalg.norm(self.quaternion)
        if norm < 1e-8:
            raise ValueError("Quaternion must be nonzero")
        self.quaternion /= norm

    def compose(self, other):
        r = rotation(self.quaternion)
        return Pose(self.position + r.apply(other.position), quaternion(r * rotation(other.quaternion)))

    def inverse(self):
        r = rotation(self.quaternion).inv()
        return Pose(r.apply(-self.position), quaternion(r))

    def to_dict(self):
        return {"position": self.position.tolist(), "quaternion_wxyz": self.quaternion.tolist()}


def pose_error(actual, target):
    return (
        float(np.linalg.norm(target.position - actual.position)),
        float((rotation(target.quaternion) * rotation(actual.quaternion).inv()).magnitude()),
    )


def interpolate_pose(start, end, fraction):
    t = float(np.clip(fraction, 0, 1))
    # Smooth velocity at both ends of a target transition.
    t = t * t * (3 - 2 * t)
    r0, r1 = rotation(start.quaternion), rotation(end.quaternion)
    r = Rotation.from_rotvec((r1 * r0.inv()).as_rotvec() * t) * r0
    return Pose((1 - t) * start.position + t * end.position, quaternion(r))


@dataclass
class ScanGoal:
    index: int
    wrist_in_object: Pose

    def __post_init__(self):
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index <= 0:
            raise ValueError("A scan goal needs a unique positive integer index")
        if not isinstance(self.wrist_in_object, Pose):
            raise TypeError("wrist_in_object must be a Pose")

    def to_dict(self):
        return {"index": self.index, "frame": "object", "right_wrist": self.wrist_in_object.to_dict()}


class PlannerWorker:
    """One outstanding next-view request, without blocking simulation/control."""

    def __init__(self, planner):
        self.planner = planner
        self.results = Queue(maxsize=1)
        self.pending = False

    def request(self, previous_scan):
        if self.pending:
            raise RuntimeError("A view-planning request is already pending")
        self.pending = True

        def work():
            try:
                goal = self.planner.next_view(previous_scan)
                if not isinstance(goal, ScanGoal):
                    raise TypeError("next_view must return a ScanGoal")
                self.results.put((goal, None))
            except Exception as exc:
                self.results.put((None, exc))

        # An unresponsive external planner must not prevent Ctrl+C shutdown.
        Thread(target=work, name="next-view-planner", daemon=True).start()

    def poll(self):
        try:
            goal, error = self.results.get_nowait()
        except Empty:
            return None
        self.pending = False
        if error is not None:
            raise RuntimeError(f"View planner failed: {error}") from error
        return goal


class RandomViewPlanner:
    """Generate ONE next view after capture, along a perturbed circular arc.

    The object frame is fixed at its centre on the ground, with +Z up. The
    wrist's +X axis aims toward the object's centre height. A bounded arc increment preserves locality;
    radius, height, pitch, and roll are independently jittered. Kinematic
    reachability is checked by the caller before sending any command.
    """

    def __init__(self, seed=7, radius=0.95, height_range=(0.82, 1.04), step_degrees=(8, 16), aim_height=0.87):
        self.rng = np.random.default_rng(seed)
        self.radius = radius
        self.height_range = height_range
        self.step_degrees = step_degrees
        self.aim_height = aim_height
        self.angle = np.pi + 0.18
        self.generated = 0

    def next_view(self, previous_scan=None):
        if self.generated and previous_scan is None:
            raise ValueError("The next view requires the previous completed scan")
        if self.generated:
            self.angle += np.deg2rad(self.rng.uniform(*self.step_degrees))
        self.generated += 1
        return self.sample_current_view()

    def sample_current_view(self):
        """Resample an unreachable proposal without advancing to a new view."""
        radius = self.radius + self.rng.uniform(-0.025, 0.025)
        height = self.rng.uniform(*self.height_range)
        yaw = self.angle + np.pi
        pitch, roll = np.deg2rad(self.rng.uniform(-4, 4, size=2))
        # Positive local Y rotation tips +X down toward the raised object.
        pitch += np.arctan2(height - self.aim_height, radius)
        r = Rotation.from_euler("ZYX", [yaw, pitch, roll])
        return ScanGoal(
            self.generated,
            Pose([radius * np.cos(self.angle), radius * np.sin(self.angle), height], quaternion(r)),
        )


def base_for_wrist(goal, forward_reach=0.32, right_offset=-0.20):
    """Place the pelvis behind and to the left of a right-wrist target."""
    p = goal.wrist_in_object.position
    yaw = np.arctan2(-p[1], -p[0])
    r = Rotation.from_euler("z", yaw)
    xy = p[:2] - r.apply([forward_reach, right_offset, 0])[:2]
    return xy, yaw


def plan_table_waypoints(
    start_xy, target_xy, obstacle_radius, clearance=0.55, spacing=0.30, max_angle_degrees=15,
):
    """Plan base positions around a circular table in the object frame.

    Interpolate radius between the two safe stances while following the shortest
    angular sweep. Equal radii give a circle; unequal radii change gradually,
    without a separate retreat or any radius larger than the endpoints. The arc
    direction is chosen once. Both points and their connecting chords clear the
    inflated table; close-to-boundary endpoints need finer angular sampling.
    These are transit waypoints, never additional wrist goals or scan requests.
    """
    start, target = np.asarray(start_xy, dtype=float), np.asarray(target_xy, dtype=float)
    if start.shape != (2,) or target.shape != (2,) or not np.isfinite([start, target]).all():
        raise ValueError("Navigation endpoints must be finite XY positions")
    parameters = [obstacle_radius, clearance, spacing, max_angle_degrees]
    if not np.isfinite(parameters).all() or min(parameters) <= 0 or max_angle_degrees > 45:
        raise ValueError("Use positive route dimensions and a waypoint angle in (0, 45] degrees")
    keepout = obstacle_radius + clearance
    r0, r1 = np.linalg.norm(start), np.linalg.norm(target)
    if min(r0, r1) <= keepout:
        raise ValueError("Navigation endpoint violates table clearance")
    a0, a1 = np.arctan2(start[1], start[0]), np.arctan2(target[1], target[0])
    sweep = wrap_angle(a1 - a0)
    # Bound each segment's polar arc length using the largest endpoint radius.
    length_bound = np.hypot(r1 - r0, max(r0, r1) * sweep)
    count = max(1, int(np.ceil(length_bound / spacing)))
    if sweep != 0:
        # A chord with endpoint radii >= r_min is outside r_min*cos(dtheta/2).
        # Preserve half the available radial margin by refining the angular
        # spacing, never by inflating the travel radius.
        r_min = min(r0, r1)
        angular_step = min(
            np.deg2rad(max_angle_degrees),
            2 * np.arccos((keepout + r_min) / (2 * r_min)),
        )
        if angular_step <= 0:
            raise ValueError("Navigation endpoints are too close to the table clearance boundary")
        count = max(count, int(np.ceil(abs(sweep) / angular_step)))
    if count > 10000:
        raise ValueError("Route needs too many waypoints; increase endpoint clearance or waypoint spacing")
    fraction = np.linspace(0, 1, count + 1)[1:]
    radii = (1 - fraction) * r0 + fraction * r1
    angles = a0 + fraction * sweep
    points = radii[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    # Preserve the caller's exact final station, including at the +/-pi seam.
    points[-1] = target.copy()
    return points


def navigation_command(base_xy, target_xy, object_radius, clearance, speed_limit=0.30):
    """Follow an arc outside a cylindrical keep-out region, with radial feedback.

    Returns velocity in the object frame. It never chooses a straight chord
    through the object, including when successive views straddle +/-pi.
    """
    base_xy, target_xy = np.asarray(base_xy), np.asarray(target_xy)
    radius = np.linalg.norm(base_xy)
    if radius <= object_radius + clearance:
        raise RuntimeError("Base entered the object's keep-out region")
    target_radius = np.linalg.norm(target_xy)
    if target_radius <= object_radius + clearance:
        raise ValueError("Requested base stance violates object clearance")
    angle = np.arctan2(base_xy[1], base_xy[0])
    target_angle = np.arctan2(target_xy[1], target_xy[0])
    radial = base_xy / radius
    tangent = np.array([-radial[1], radial[0]])
    velocity = 0.9 * (target_radius - radius) * radial
    velocity += 0.9 * radius * wrap_angle(target_angle - angle) * tangent
    norm = np.linalg.norm(velocity)
    if norm > speed_limit:
        velocity *= speed_limit / norm
    return velocity


@dataclass
class ArrivalGate:
    position_tolerance: float = 0.05
    orientation_tolerance: float = np.deg2rad(10)
    linear_speed_tolerance: float = 0.06
    angular_speed_tolerance: float = 0.25
    hold_seconds: float = 0.75
    since: float | None = None

    def update(self, now, actual, target, linear_speed, angular_speed, base_speed):
        position_error, orientation_error = pose_error(actual, target)
        inside = (
            position_error <= self.position_tolerance
            and orientation_error <= self.orientation_tolerance
            and linear_speed <= self.linear_speed_tolerance
            and angular_speed <= self.angular_speed_tolerance
            and base_speed <= self.linear_speed_tolerance
        )
        if not inside:
            self.since = None
        elif self.since is None:
            self.since = now
        return self.since is not None and now - self.since >= self.hold_seconds


# These are SONIC's virtual tracking-point offsets, NOT the scanner extrinsic.
# Bodies: left_wrist_yaw_link, right_wrist_yaw_link, torso_link.
TRACKING_OFFSETS = ([0.18, -0.025, 0], [0.18, 0.025, 0], [0, 0, 0.35])


def three_point_targets(base_world, body_targets_world):
    """Convert link-frame targets to SONIC's root-relative virtual trackers."""
    local = []
    inverse_base = base_world.inverse()
    for body, offset in zip(body_targets_world, TRACKING_OFFSETS):
        tracker = body.compose(Pose(offset, [1, 0, 0, 0]))
        local.append(inverse_base.compose(tracker))
    return np.concatenate([p.position for p in local]), np.concatenate([p.quaternion for p in local])
