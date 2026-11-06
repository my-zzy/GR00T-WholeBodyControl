"""Geometry, sequencing, arrival, and real MuJoCo ray-capture regressions.

Run: python -m unittest gear_sonic.examples.object_scan.test_object_scan
"""

from pathlib import Path
import tempfile
from threading import Event
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    plan_table_waypoints,
    pose_error,
    quaternion,
    three_point_targets,
)
from gear_sonic.examples.object_scan.run import Config, ScanSession
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

    def test_far_side_waypoints_and_segments_clear_table(self):
        # Include opposite sides, +/-pi wrapping, unequal radii, and larger tables.
        for a0, a1, r0, r1, obstacle in [
            (0, 180, 1.35, 1.28, 0.45), (0, -170, 1.35, 1.28, 0.45),
            (0, 170, 1.35, 1.28, 0.45), (0, 0, 1.35, 1.35, 0.45),
            (179, -179, 1.28, 1.28, 0.45), (0, 170, 1.01, 1.02, 0.45),
            (20, -150, 2.8, 2.6, 2.0), (20, 20, 1.35, 1.05, 0.45),
        ]:
            with self.subTest(angles=(a0, a1), obstacle=obstacle):
                start = r0 * np.array([np.cos(np.deg2rad(a0)), np.sin(np.deg2rad(a0))])
                target = r1 * np.array([np.cos(np.deg2rad(a1)), np.sin(np.deg2rad(a1))])
                path = plan_table_waypoints(start, target, obstacle)
                np.testing.assert_array_equal(path[-1], target)
                points = np.vstack([start, path])
                for begin, end in zip(points[:-1], points[1:]):
                    segment = end - begin
                    length = np.linalg.norm(segment)
                    self.assertLessEqual(length, 0.30 + 1e-9)
                    t = np.clip(-begin.dot(segment) / segment.dot(segment), 0, 1) if length else 0
                    self.assertGreater(np.linalg.norm(begin + t * segment), obstacle + 0.55)
                if abs(a1 - a0) == 180:
                    self.assertGreater(len(path), 10)
                if (a0, a1) == (179, -179):
                    self.assertEqual(len(path), 1)

    def test_waypoint_route_rejects_unsafe_or_invalid_endpoints(self):
        for start, target in [([0.9, 0], [1.3, 0]), ([1.3, 0], [0.9, 0]),
                              ([float("nan"), 0], [1.3, 0])]:
            with self.assertRaises(ValueError):
                plan_table_waypoints(start, target, 0.45)
        for kwargs in ({"spacing": 0}, {"clearance": -1}, {"max_angle_degrees": 90}):
            with self.assertRaises(ValueError):
                plan_table_waypoints([1.3, 0], [-1.3, 0], 0.45, **kwargs)

    def test_waypoints_do_not_add_outward_retreat(self):
        for r0, r1 in [(1.35, 1.25), (1.25, 1.35), (1.05, 1.05)]:
            with self.subTest(radii=(r0, r1)):
                start = np.array([r0, 0])
                angle = np.deg2rad(170)
                target = r1 * np.array([np.cos(angle), np.sin(angle)])
                path = plan_table_waypoints(start, target, 0.45)
                radii = np.linalg.norm(np.vstack([start, path]), axis=1)
                self.assertTrue(np.all(radii <= max(r0, r1) + 1e-12))
                self.assertTrue(np.all(radii >= min(r0, r1) - 1e-12))
                if r1 <= r0:
                    self.assertTrue(np.all(np.diff(radii) <= 1e-12))
                else:
                    self.assertTrue(np.all(np.diff(radii) >= -1e-12))
                # Even the first waypoint makes angular progress: no retreat leg.
                self.assertGreater(np.arctan2(path[0, 1], path[0, 0]), 0)

    def test_near_boundary_arc_refines_chords_without_increasing_radius(self):
        radius = 1.0001
        start = np.array([radius, 0])
        path = plan_table_waypoints(start, -start, 0.45)
        np.testing.assert_allclose(np.linalg.norm(path, axis=1), radius, atol=1e-12)
        points = np.vstack([start, path])
        midpoints = (points[:-1] + points[1:]) / 2
        self.assertTrue(np.all(np.linalg.norm(midpoints, axis=1) > 1.0))

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
        for kwargs in ({"table_clearance": 0.1}, {"waypoint_spacing": 0.05},
                       {"waypoint_angle_degrees": 90}, {"step_max_degrees": 181}):
            with self.assertRaises(ValueError):
                Config(**kwargs).validate()
        Config(step_min_degrees=180, step_max_degrees=180).validate()

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


class WaypointSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Pose([-1.35, 0, 0.79], [1, 0, 0, 0])
        self.carry = [Pose([0.32, 0.2, 0.1], [1, 0, 0, 0]),
                      Pose([0.32, -0.2, 0.1], [1, 0, 0, 0]),
                      Pose([0, 0, 0.5], [1, 0, 0, 0])]
        self.wrist_override = None
        sim = SimpleNamespace(
            object_pose=Pose([0, 0, 0], [1, 0, 0, 0]), data=SimpleNamespace(time=0),
            reachable=lambda *args: (True, 0, 0), set_target=Mock(),
            body_speed=lambda name: (0, 0), capture=Mock(side_effect=AssertionError("Unexpected capture")),
        )
        sim.body_pose = lambda name: (self.base if name == "pelvis" else
                                     self.wrist_override or self.base.compose(self.carry[1]))
        self.controller = Mock()
        self.session = ScanSession(Config(), sim, self.controller, Path(self.directory.name))
        self.addCleanup(self.session.telemetry.close)
        self.session.nominal_local = self.carry
        self.goal = ScanGoal(1, Pose([0.95, 0, 0.9], quaternion(Rotation.from_euler("z", np.pi))))
        self.session.accept_goal(self.goal)

    def test_transit_waypoints_do_not_become_scan_or_wrist_targets(self):
        session = self.session
        self.assertGreater(len(session.navigation_waypoints), 10)
        original_wrist = session.wrist_target.to_dict()
        for i, xy in enumerate(session.navigation_waypoints):
            yaw = np.arctan2(-xy[1], -xy[0])
            self.base = Pose([*xy, 0.79], quaternion(Rotation.from_euler("z", yaw)))
            session.tick()
            self.assertEqual(session.wrist_target.to_dict(), original_wrist)
            if np.linalg.norm(xy - session.base_target) > 0.09:
                self.assertEqual(session.phase, "move")
                self.assertEqual(session.waypoint_index, i + 1)
                self.assertEqual(self.controller.send_planner.call_args.args[0], 1)
                sent_positions = self.controller.send_planner.call_args.args[4][0]
                carry_positions = three_point_targets(self.base, [self.base.compose(p) for p in self.carry])[0]
                np.testing.assert_allclose(sent_positions, carry_positions)
            else:
                self.assertEqual(session.waypoint_index, len(session.navigation_waypoints) - 1)
        self.assertEqual(session.phase, "settle")
        session.sim.capture.assert_not_called()
        self.assertIs(session.goal, self.goal)
        self.assertTrue((Path(self.directory.name) / "route_001.json").is_file())

    def test_next_goal_waits_for_measured_arm_retraction(self):
        session = self.session
        session.phase, session.phase_start, session.goal_start = "retract", 90, 90
        session.return_start = session.wrist_target
        session.next_goal = Mock()
        self.wrist_override = session.wrist_target
        with patch("gear_sonic.examples.object_scan.run.time.monotonic", return_value=100):
            session.tick()
        session.next_goal.assert_not_called()
        self.wrist_override = None
        with patch("gear_sonic.examples.object_scan.run.time.monotonic", return_value=101):
            session.tick()
        session.next_goal.assert_not_called()
        with patch("gear_sonic.examples.object_scan.run.time.monotonic", return_value=101.31):
            session.tick()
        session.next_goal.assert_called_once()


if __name__ == "__main__":
    unittest.main()
