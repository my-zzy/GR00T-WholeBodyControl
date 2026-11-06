"""Geometry, sequencing, arrival, and real MuJoCo ray-capture regressions.

Run: python -m unittest gear_sonic.examples.object_scan.test_object_scan
"""

from pathlib import Path
import tempfile
from threading import Event
import time
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.examples.object_scan.planning import (
    ArrivalGate,
    PlannerWorker,
    Pose,
    RandomViewPlanner,
    ScanGoal,
    base_for_wrist,
    interpolate_pose,
    navigation_command,
    pose_error,
    quaternion,
    three_point_targets,
)
from gear_sonic.examples.object_scan.run import Config
from gear_sonic.examples.object_scan.simulation import ScanSimulation


class ScanGeometryTests(unittest.TestCase):
    def test_root_relative_targets_preserve_world_pose_and_virtual_offsets(self):
        base = Pose([3, 4, 0.8], quaternion(Rotation.from_euler("z", np.pi / 2)))
        wrists = [Pose([4, 5, 1], [1, 0, 0, 0])] * 3
        positions, orientations = three_point_targets(base, wrists)
        # The right-wrist tracker is displaced +0.18 X,+0.025 Y in wrist axes.
        actual_world = base.compose(Pose(positions[3:6], orientations[4:8]))
        np.testing.assert_allclose(actual_world.position, [4.18, 5.025, 1], atol=1e-9)
        self.assertLess(pose_error(actual_world, Pose([4.18, 5.025, 1], [1, 0, 0, 0]))[1], 1e-9)

    def test_orientation_interpolation_takes_short_path_across_pi(self):
        a = Pose([0, 0, 0], quaternion(Rotation.from_euler("z", np.deg2rad(179))))
        b = Pose([1, 0, 0], quaternion(Rotation.from_euler("z", np.deg2rad(-179))))
        middle = interpolate_pose(a, b, 0.5)
        self.assertLess(pose_error(a, middle)[1], np.deg2rad(1.01))
        np.testing.assert_allclose(middle.position, [0.5, 0, 0])

    def test_arc_navigation_does_not_take_chord_through_object(self):
        velocity = navigation_command([1.2, 0], [-1.2, 0], 0.6, 0.35)
        self.assertAlmostEqual(velocity[0], 0)
        self.assertAlmostEqual(np.linalg.norm(velocity), 0.3)
        with self.assertRaises(RuntimeError):
            navigation_command([0.8, 0], [1.2, 0], 0.6, 0.35)

    def test_arc_navigation_wraps_angles(self):
        a, b = np.deg2rad([179, -179])
        current = 1.2 * np.array([np.cos(a), np.sin(a)])
        target = 1.2 * np.array([np.cos(b), np.sin(b)])
        velocity = navigation_command(current, target, 0.6, 0.35)
        self.assertLess(np.linalg.norm(velocity), 0.05)
        self.assertGreater(np.dot(velocity, [-current[1], current[0]]), 0)

    def test_generator_waits_for_scan_and_stays_within_sampling_bounds(self):
        planner = RandomViewPlanner(seed=7)
        goal = planner.next_view()
        with self.assertRaises(ValueError):
            planner.next_view()
        heights = []
        for _ in range(30):
            goal = planner.next_view({"capture_complete": True})
            p = goal.wrist_in_object.position
            heights.append(p[2])
            self.assertTrue(0.925 <= np.linalg.norm(p[:2]) <= 0.975)
            self.assertTrue(0.82 <= p[2] <= 1.04)
            base, _ = base_for_wrist(goal)
            self.assertGreater(np.linalg.norm(base), 1.1)
        self.assertGreater(np.ptp(heights), 0.15)

    def test_arrival_needs_continuous_pose_and_velocity_compliance(self):
        target = Pose([0, 0, 1], [1, 0, 0, 0])
        gate = ArrivalGate(hold_seconds=0.5)
        self.assertFalse(gate.update(0, target, target, 0, 0, 0))
        self.assertFalse(gate.update(0.4, target, target, 0.5, 0, 0))
        self.assertFalse(gate.update(0.6, target, target, 0, 0, 0))
        self.assertFalse(gate.update(1.0, target, target, 0, 0, 0))
        self.assertTrue(gate.update(1.2, target, target, 0, 0, 0))
        rotated = Pose(target.position, quaternion(Rotation.from_euler("z", 0.5)))
        self.assertFalse(gate.update(1.3, rotated, target, 0, 0, 0))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            Pose([0, 0, 0], [0, 0, 0, 0])
        with self.assertRaises(ValueError):
            Pose([float("nan"), 0, 0], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            Config(wrist_radius=0.7).validate()
        with self.assertRaises(ValueError):
            Config(height_max=1.8).validate()
        with self.assertRaises(ValueError):
            Config(speed=0.15).validate()
        with self.assertRaises(ValueError):
            Config(position_tolerance=float("nan")).validate()
        with self.assertRaises(ValueError):
            ScanGoal(0, Pose([1, 0, 1], [1, 0, 0, 0]))

    def test_slow_planner_does_not_block_control_or_accept_concurrent_requests(self):
        released = Event()
        previous = {"path": "actual_scan.npz"}

        class SlowPlanner:
            def next_view(self, scan):
                self.received = scan
                released.wait(timeout=2)
                return ScanGoal(2, Pose([1, 0, 1], [1, 0, 0, 0]))

        planner = SlowPlanner()
        worker = PlannerWorker(planner)
        try:
            worker.request(previous)
            self.assertIsNone(worker.poll())
            with self.assertRaises(RuntimeError):
                worker.request(previous)
        finally:
            released.set()
        deadline = time.monotonic() + 2
        result = None
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(0.001)
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 2)
        self.assertIs(planner.received, previous)

    def test_planner_failure_is_reported_instead_of_a_fake_goal(self):
        class BrokenPlanner:
            def next_view(self, scan):
                raise ValueError("unreachable request")

        worker = PlannerWorker(BrokenPlanner())
        worker.request(None)
        deadline = time.monotonic() + 2
        with self.assertRaisesRegex(RuntimeError, "unreachable request"):
            while time.monotonic() < deadline:
                worker.poll()
                time.sleep(0.001)

    def test_capture_intersects_actual_cylinder_and_applies_scanner_extrinsic(self):
        sim = ScanSimulation.__new__(ScanSimulation)
        sim.model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom name="object" type="cylinder" size="0.6 0.7" pos="0 0 0.7"/>
          <body name="right_wrist_yaw_link" pos="-1.22 -0.025 0.8">
            <body name="virtual_scanner" pos="0.22 0.025 0"/>
          </body></worldbody></mujoco>""")
        sim.data = mujoco.MjData(sim.model)
        mujoco.mj_forward(sim.model, sim.data)
        sim.object_pose = Pose([0, 0, 0], [1, 0, 0, 0])
        sim.object_geom = sim.model.geom("object").id
        sim.scanner_body = sim.model.body("virtual_scanner").id
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "capture.npz"
            result = sim.capture(path, width=12, height=10)
            self.assertGreater(result["point_count"], 20)
            saved = np.load(path)
            np.testing.assert_allclose(saved["scanner_position"], [-1, 0, 0.8], atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(saved["points_object"][:, :2], axis=1), 0.6, atol=1e-7)
            self.assertTrue(path.with_suffix(".ply").exists())


if __name__ == "__main__":
    unittest.main()
