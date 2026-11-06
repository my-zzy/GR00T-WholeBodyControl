"""Run an adaptive, closed-loop SONIC v1.1 scan in MuJoCo.

    conda activate sonic
    python -m gear_sonic.examples.object_scan.run --views 3

The executable and all SDK packages must already be installed. This launcher
only controls a simulator on loopback; it has no physical-robot mode.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import pty
import signal
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation
import zmq

from gear_sonic.examples.object_scan.planning import (
    ArrivalGate,
    PlannerWorker,
    Pose,
    RandomViewPlanner,
    base_for_wrist,
    interpolate_pose,
    navigation_command,
    pose_error,
    quaternion,
    rotation,
    three_point_targets,
    wrap_angle,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Config:
    views: int = 3
    seed: int = 7
    headless: bool = False
    record: bool = False
    output: str = ""
    object_radius: float = 0.18
    object_height: float = 0.30
    table_radius: float = 0.45
    table_height: float = 0.72
    object_distance: float = 1.35
    wrist_radius: float = 0.95
    height_min: float = 0.82
    height_max: float = 1.04
    step_min_degrees: float = 8
    step_max_degrees: float = 16
    scanner_mass: float = 0.15
    position_tolerance: float = 0.05
    orientation_tolerance_degrees: float = 10
    hold_seconds: float = 0.75
    speed: float = 0.30
    view_timeout: float = 150
    startup_timeout: float = 600
    command_port: int = 5566
    feedback_port: int = 5567
    planner: str = ""

    @property
    def obstacle_radius(self):
        return max(self.object_radius, self.table_radius)

    def validate(self):
        if any(isinstance(value, (int, float)) and not np.isfinite(value) for value in vars(self).values()):
            raise ValueError("Configuration values must be finite")
        if self.views < 1 or not 0.65 <= self.height_min <= self.height_max <= 1.20:
            raise ValueError("Use views >= 1 and wrist heights within [0.65, 1.20] m")
        if self.object_radius <= 0 or self.object_height <= 0 or self.scanner_mass <= 0:
            raise ValueError("Object dimensions and scanner mass must be positive")
        if self.table_radius < self.object_radius + 0.03 or self.table_height <= 0.06:
            raise ValueError("The table must fit the object with >= 0.03 m margin and be > 0.06 m high")
        if self.height_min < self.table_height + 0.08:
            raise ValueError("Wrist heights must be at least 0.08 m above the tabletop")
        if self.wrist_radius < self.obstacle_radius + 0.30:
            raise ValueError("Wrist radius must leave >= 0.30 m outside the object/table")
        if self.object_distance < self.obstacle_radius + 0.55:
            raise ValueError("Initial robot stance is too close to the object")
        if not 0.2 <= self.speed <= 0.4 or not 0 < self.step_min_degrees <= self.step_max_degrees <= 45:
            raise ValueError("Use slow-walk speed in [0.2, 0.4] m/s and arc steps in (0, 45] degrees")
        if (
            min(
                self.position_tolerance,
                self.orientation_tolerance_degrees,
                self.hold_seconds,
                self.view_timeout,
                self.startup_timeout,
            )
            <= 0
        ):
            raise ValueError("Tolerances and timeouts must be positive")
        if self.command_port == self.feedback_port or not all(
            1024 <= p <= 65535 for p in (self.command_port, self.feedback_port)
        ):
            raise ValueError("Choose two distinct unprivileged TCP ports")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def preflight():
    deploy = ROOT / "gear_sonic_deploy"
    required = [
        "target/release/g1_deploy_onnx_ref",
        "policy/sonic_v1_1/model_encoder.onnx",
        "policy/sonic_v1_1/model_decoder.onnx",
        "policy/sonic_v1_1/observation_config.yaml",
        "planner/target_vel/V2/planner_sonic.onnx",
    ]
    for name in required:
        p = deploy / name
        if not p.is_file():
            raise FileNotFoundError(f"Required deployment file is missing: {p}")
        with p.open("rb") as f:
            if f.read(100).startswith(b"version https://git-lfs.github.com/spec"):
                raise RuntimeError(f"Download the actual model, not its Git LFS pointer: {p}")
    # The upstream binary uses DDS domain 0. Do not share it with another run.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if args and (
            Path(os.fsdecode(args[0])).name == "g1_deploy_onnx_ref"
            or any(a.endswith(b"/run_sim_loop.py") for a in args[:3])
        ):
            raise RuntimeError(f"Another SONIC controller/simulator is running (PID {entry.name}); stop it first")


class ControllerProcess:
    def __init__(self, config, output):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.publisher.bind(f"tcp://127.0.0.1:{config.command_port}")
        self.master, slave = pty.openpty()
        self.log = (output / "controller.log").open("w")
        self.reader = (output / "controller.log").open()
        self.text = ""
        self.ready = False
        self.controlling = False
        args = [
            "./target/release/g1_deploy_onnx_ref",
            "lo",
            "policy/sonic_v1_1/model_decoder.onnx",
            "reference/example",
            "--encoder-file",
            "policy/sonic_v1_1/model_encoder.onnx",
            "--obs-config",
            "policy/sonic_v1_1/observation_config.yaml",
            "--planner-file",
            "planner/target_vel/V2/planner_sonic.onnx",
            "--motor-kp-scale",
            "4,10=1.5",
            "--motor-kd-scale",
            "4,10=1.5",
            "--input-type",
            "zmq_manager",
            "--zmq-host",
            "127.0.0.1",
            "--zmq-port",
            str(config.command_port),
            "--output-type",
            "zmq",
            "--zmq-out-port",
            str(config.feedback_port),
            "--disable-crc-check",
        ]
        try:
            self.process = subprocess.Popen(
                args,
                cwd=ROOT / "gear_sonic_deploy",
                stdin=slave,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            os.close(slave)

    def poll(self):
        self.text = (self.text + self.reader.read())[-65536:]
        self.ready |= "Init Done" in self.text
        self.controlling |= "transitioning to CONTROL state" in self.text
        if self.process.poll() is not None:
            raise RuntimeError(f"SONIC controller exited ({self.process.returncode}); inspect controller.log")

    def send_command(self, start=False, stop=False):
        self.publisher.send(build_command_message(start=start, stop=stop, planner=True))

    def send_planner(self, mode, movement, facing, speed=-1, targets=None):
        kwargs = {}
        if targets is not None:
            kwargs.update(vr_3pt_position=targets[0], vr_3pt_orientation=targets[1])
        self.publisher.send(build_planner_message(mode, movement, facing, speed=speed, **kwargs))

    def close(self):
        if hasattr(self, "process") and self.process.poll() is None:
            self.send_command(stop=True)
            os.write(self.master, b"o")
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGINT)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        self.publisher.close()
        self.context.term()
        self.log.close()
        self.reader.close()
        os.close(self.master)


class ScanSession:
    def __init__(self, config, sim, controller, output, planner=None):
        self.config, self.sim, self.controller, self.output = config, sim, controller, output
        self.object_pose = sim.object_pose
        self.planner = planner or RandomViewPlanner(
            config.seed,
            config.wrist_radius,
            (config.height_min, config.height_max),
            (config.step_min_degrees, config.step_max_degrees),
            aim_height=config.table_height + config.object_height / 2,
        )
        self.planner_worker = PlannerWorker(self.planner)
        self.phase = "initializing"
        self.phase_start = time.monotonic()
        self.started = self.phase_start
        self.last_report = 0
        self.last_start = 0
        self.scans = []
        self.goal = None
        self.nominal_local = None
        self.ready_since = None
        self.gate = ArrivalGate(
            position_tolerance=config.position_tolerance,
            orientation_tolerance=np.deg2rad(config.orientation_tolerance_degrees),
            hold_seconds=config.hold_seconds,
        )
        self.position_correction = np.zeros(3)
        self.rotation_correction = np.zeros(3)
        self.last_navigation_update = -float("inf")
        self.navigation_direction = np.zeros(2)
        self.navigation_facing = np.array([1.0, 0.0, 0.0])
        self.telemetry = (output / "telemetry.jsonl").open("w", buffering=1)

    def transition(self, phase):
        self.phase = phase
        self.phase_start = time.monotonic()
        self.gate.since = None
        print(f"[{self.sim.data.time:7.2f}s] {phase}", flush=True)

    def next_goal(self):
        previous = self.scans[-1] if self.scans else None
        self.planner_worker.request(previous)
        self.transition("waiting for view")

    def accept_goal(self, goal):
        if any(scan["goal"]["index"] == goal.index for scan in self.scans):
            raise ValueError(f"View planner reused scan index {goal.index}")
        self.goal = goal
        for attempt in range(30):
            self.base_target, self.base_yaw = base_for_wrist(self.goal)
            self.wrist_target = self.object_pose.compose(self.goal.wrist_in_object)
            base_world = self.object_pose.compose(Pose([*self.base_target, 0], [1, 0, 0, 0])).position[:2]
            reachable, pos_error, angle_error = self.sim.reachable(self.wrist_target, base_world, self.base_yaw)
            if reachable and np.linalg.norm(self.base_target) > self.config.obstacle_radius + 0.40:
                break
            if not hasattr(self.planner, "sample_current_view"):
                raise RuntimeError("External planner supplied an unreachable wrist pose")
            self.goal = self.planner.sample_current_view()
        else:
            raise RuntimeError(
                f"No reachable goal after 30 proposals; IK error {pos_error:.3f}m/{np.rad2deg(angle_error):.1f}deg"
            )
        self.goal_start = time.monotonic()
        self.last_navigation_update = -float("inf")
        self.position_correction[:] = 0
        self.rotation_correction[:] = 0
        self.sim.set_target(self.wrist_target)
        write_json(self.output / f"goal_{self.goal.index:03d}.json", self.goal.to_dict())
        self.transition("move")
        print(
            f"  View {self.goal.index}: object-relative wrist {self.goal.wrist_in_object.position.round(3)}",
            flush=True,
        )

    def tick(self):
        now = time.monotonic()
        self.controller.poll()
        base = self.sim.body_pose("pelvis")
        base_obj = self.object_pose.inverse().compose(base)
        base_xy = base_obj.position[:2]
        facing = -base_xy / np.linalg.norm(base_xy)
        facing_world = rotation(self.object_pose.quaternion).apply([*facing, 0])
        velocity = np.zeros(2)
        mode, speed, targets = 0, -1, None
        actual_wrist = self.sim.body_pose("right_wrist_yaw_link")

        if self.phase == "initializing":
            if now - self.started > self.config.startup_timeout:
                raise TimeoutError("Controller startup timed out; inspect controller.log")
            if self.controller.ready and not self.controller.controlling and now - self.last_start > 1:
                self.controller.send_command(start=True)
                self.last_start = now
            if self.controller.controlling:
                self.transition("release")
        elif self.phase == "release":
            if now - self.phase_start > 1.5:
                self.sim.env.elastic_band.enable = False
                self.transition("stand")
        elif self.phase == "stand":
            if now - self.phase_start > 4:
                inverse = base.inverse()
                self.nominal_local = [inverse.compose(p) for p in self.sim.tracking_poses()]
                self.next_goal()
        elif self.phase == "waiting for view":
            targets = three_point_targets(base, [base.compose(p) for p in self.nominal_local])
            if now - self.phase_start > self.config.view_timeout:
                raise TimeoutError("View planner did not supply a goal before the timeout")
            goal = self.planner_worker.poll()
            if goal is not None:
                self.accept_goal(goal)
        else:
            if now - self.goal_start > self.config.view_timeout:
                error = pose_error(actual_wrist, self.wrist_target)
                raise TimeoutError(
                    f"View {self.goal.index} timed out in {self.phase}: wrist error "
                    f"{error[0]:.3f}m/{np.rad2deg(error[1]):.1f}deg"
                )
            nominal = [base.compose(p) for p in self.nominal_local]
            if self.phase == "move":
                velocity = navigation_command(
                    base_xy, self.base_target, self.config.obstacle_radius, 0.35, self.config.speed
                )
                distance = np.linalg.norm(base_xy - self.base_target)
                yaw = rotation(base.quaternion).as_euler("ZYX")[0]
                heading_error = abs(wrap_angle(np.arctan2(facing_world[1], facing_world[0]) - yaw))
                if distance > 0.09:
                    # SLOW_WALK needs >= 0.2 m/s. Scaling speed all the way to
                    # zero can leave the learned planner standing indefinitely.
                    mode, speed = 1, self.config.speed
                    # The native planner replans on ANY direction float change.
                    # Hold navigation commands briefly while still publishing
                    # fresh tracking targets and the command heartbeat at 50 Hz.
                    if now - self.last_navigation_update >= 0.5:
                        self.navigation_direction = velocity.copy()
                        self.navigation_facing = facing_world.copy()
                        self.last_navigation_update = now
                    velocity = self.navigation_direction
                    facing_world = self.navigation_facing
                else:
                    velocity[:] = 0
                    if heading_error < np.deg2rad(12):
                        self.transition("settle")
                targets = three_point_targets(base, nominal)
            elif self.phase == "settle":
                targets = three_point_targets(base, nominal)
                if now - self.phase_start > 2 and self.sim.body_speed("pelvis")[0] < 0.10:
                    self.wrist_start = actual_wrist
                    self.transition("position wrist")
            elif self.phase == "position wrist":
                fraction = (now - self.phase_start) / 3.0
                desired = interpolate_pose(self.wrist_start, self.wrist_target, fraction)
                if fraction >= 1:
                    # Small, bounded integral bias correction; acceptance always
                    # checks the ORIGINAL target, not this corrected command.
                    self.position_correction += 0.20 * 0.02 * (self.wrist_target.position - actual_wrist.position)
                    norm = np.linalg.norm(self.position_correction)
                    if norm > 0.10:
                        self.position_correction *= 0.10 / norm
                    angular_error = (
                        rotation(self.wrist_target.quaternion) * rotation(actual_wrist.quaternion).inv()
                    ).as_rotvec()
                    self.rotation_correction += 0.10 * 0.02 * angular_error
                    norm = np.linalg.norm(self.rotation_correction)
                    if norm > 0.30:
                        self.rotation_correction *= 0.30 / norm
                desired = Pose(
                    desired.position + self.position_correction,
                    quaternion(Rotation.from_rotvec(self.rotation_correction) * rotation(desired.quaternion)),
                )
                nominal[1] = desired
                targets = three_point_targets(base, nominal)
                wrist_speed, wrist_angular_speed = self.sim.body_speed("right_wrist_yaw_link")
                base_speed = self.sim.body_speed("pelvis")[0]
                if fraction >= 1 and self.gate.update(
                    now, actual_wrist, self.wrist_target, wrist_speed, wrist_angular_speed, base_speed
                ):
                    result = self.sim.capture(self.output / f"scan_{self.goal.index:03d}.npz")
                    position_error, orientation_error = pose_error(actual_wrist, self.wrist_target)
                    result.update(
                        goal=self.goal.to_dict(),
                        position_error_m=position_error,
                        orientation_error_degrees=float(np.rad2deg(orientation_error)),
                        sim_time=float(self.sim.data.time),
                    )
                    write_json(self.output / f"scan_{self.goal.index:03d}.json", result)
                    self.scans.append(result)
                    print(
                        f"  CAPTURED view {self.goal.index}: {result['point_count']} points, "
                        f"error {position_error:.3f}m/{np.rad2deg(orientation_error):.1f}deg",
                        flush=True,
                    )
                    if len(self.scans) >= self.config.views:
                        return True
                    # Return to a comfortable carrying pose smoothly before walking.
                    self.return_start = actual_wrist
                    self.transition("retract")
            elif self.phase == "retract":
                nominal[1] = interpolate_pose(self.return_start, nominal[1], (now - self.phase_start) / 2)
                targets = three_point_targets(base, nominal)
                if now - self.phase_start > 2.5:
                    self.next_goal()
            if self.goal:
                position_error, orientation_error = pose_error(actual_wrist, self.wrist_target)
                record = {
                    "time": float(self.sim.data.time),
                    "view": self.goal.index,
                    "phase": self.phase,
                    "base_xy_object": base_xy.tolist(),
                    "base_target_xy_object": self.base_target.tolist(),
                    "command_mode": mode,
                    "command_speed": speed,
                    "wrist_world": actual_wrist.to_dict(),
                    "position_error_m": position_error,
                    "orientation_error_degrees": float(np.rad2deg(orientation_error)),
                    "wrist_speed": self.sim.body_speed("right_wrist_yaw_link")[0],
                    "base_speed": self.sim.body_speed("pelvis")[0],
                }
                self.telemetry.write(json.dumps(record, allow_nan=False) + "\n")
        direction_world = rotation(self.object_pose.quaternion).apply([*velocity, 0])
        self.controller.send_planner(mode, direction_world, facing_world, speed, targets)
        if now - self.last_report > 5:
            suffix = ""
            if self.goal:
                pe, re = pose_error(actual_wrist, self.wrist_target)
                suffix = f", wrist error {pe:.3f}m/{np.rad2deg(re):.1f}deg"
            print(f"[{self.sim.data.time:7.2f}s] {self.phase}{suffix}", flush=True)
            self.last_report = now
        return False


def main(config):
    config.validate()
    output = (
        Path(config.output).expanduser().resolve()
        if config.output
        else (ROOT / ".cache/object_scan" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    lock_path = ROOT / ".cache/object_scan/run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    sim = controller = session = recording = None
    recording_path = None
    status, error = "failed", None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight()
        if config.headless:
            os.environ.setdefault("MUJOCO_GL", "egl")
        from gear_sonic.examples.object_scan.simulation import ScanSimulation

        object_pose = Pose([config.object_distance, 0, 0], [1, 0, 0, 0])
        sim = ScanSimulation(
            output,
            object_pose,
            config.object_radius,
            config.object_height,
            config.scanner_mass,
            config.headless,
            table_radius=config.table_radius,
            table_height=config.table_height,
        )
        controller = ControllerProcess(config, output)
        planner = None
        if config.planner:
            module, name = config.planner.split(":", 1)
            planner = getattr(importlib.import_module(module), name)(config)
        session = ScanSession(config, sim, controller, output, planner)
        if config.record:
            from gear_sonic.examples.object_scan.recording import StateRecording

            recording = StateRecording()
        print(f"Simulation-only scan; logs and captures: {output}", flush=True)
        print(
            "Close the viewer or press Ctrl+C to stop. First v1.1 startup may build TensorRT engines.", flush=True
        )
        step = 0
        while True:
            begin = time.monotonic()
            sim.step()
            if step % 4 == 0:
                completed = session.tick()
                if recording:
                    recording.append(sim, session)
                if completed:
                    status = "complete"
                    break
                if sim.env.viewer:
                    if not sim.env.viewer.is_running():
                        raise KeyboardInterrupt
                    sim.env.update_viewer()
            step += 1
            time.sleep(max(0, sim.dt - (time.monotonic() - begin)))
    except KeyboardInterrupt:
        status, error = "interrupted", "User stopped the simulation"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, flush=True)
    finally:
        if controller:
            controller.close()
        if session:
            session.telemetry.close()
        if sim:
            sim.close()
        if recording:
            recording_path = recording.save(output)
        summary = {
            "status": status,
            "error": error,
            "requested_views": config.views,
            "completed_views": len(session.scans) if session else 0,
            "scans": session.scans if session else [],
            "recording": recording_path,
        }
        write_json(output / "summary.json", summary)
        lock.close()
        print(f"Result: {status}. See {output / 'summary.json'}", flush=True)
    return 0 if status == "complete" else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, field in Config.__dataclass_fields__.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(flag, action="store_true", default=field.default)
        else:
            parser.add_argument(flag, type=type(field.default), default=field.default)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
