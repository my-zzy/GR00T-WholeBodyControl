"""MuJoCo scene, ground-truth feedback, reach checks, and a ray-based scanner."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.examples.object_scan.planning import Pose, pose_error, quaternion, rotation
from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge

ROOT = Path(__file__).resolve().parents[3]
WRIST_BODY = "right_wrist_yaw_link"
TRACKING_BODIES = ("left_wrist_yaw_link", WRIST_BODY, "torso_link")
SCANNER_IN_WRIST = Pose([0.22, 0.025, 0], [1, 0, 0, 0])


def make_scene(
    path, object_pose, object_radius, object_height, scanner_mass, table_radius=0.45, table_height=0.72
):
    source = ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
    spec = mujoco.MjSpec.from_file(str(source))
    spec.meshdir = str(source.parent / "meshes")
    table = spec.worldbody.add_body(name="scan_table", pos=object_pose.position)
    top_thickness = 0.06
    table.add_geom(
        name="scan_table_top",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, table_height - top_thickness / 2],
        size=[table_radius, top_thickness / 2, 0],
        rgba=[0.48, 0.30, 0.16, 1],
        contype=1,
        conaffinity=1,
    )
    leg_height = table_height - top_thickness
    leg_radius = min(0.035, table_radius * 0.1)
    for index, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        table.add_geom(
            name=f"scan_table_leg_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x * table_radius * 0.55, y * table_radius * 0.55, leg_height / 2],
            size=[leg_radius, leg_height / 2, 0],
            rgba=[0.22, 0.24, 0.26, 1],
            contype=1,
            conaffinity=1,
        )
    obj = spec.worldbody.add_body(name="scan_object", pos=object_pose.position + [0, 0, table_height])
    obj.add_geom(
        name="scan_object_surface",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, object_height / 2],
        size=[object_radius, object_height / 2, 0],
        rgba=[0.55, 0.67, 0.76, 1],
        contype=1,
        conaffinity=1,
    )
    scanner = spec.body(WRIST_BODY).add_body(name="virtual_scanner", pos=SCANNER_IN_WRIST.position)
    scanner.add_geom(
        name="scanner_housing",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.02],
        mass=scanner_mass,
        rgba=[0.1, 0.8, 0.8, 1],
        contype=0,
        conaffinity=0,
    )
    scanner.add_site(name="scanner_optical_frame", size=[0.01, 0, 0], rgba=[0, 1, 1, 1])
    marker = spec.worldbody.add_body(name="scan_target", mocap=True, pos=[0, 0, -10])
    marker.add_site(name="scan_target_position", size=[0.02, 0, 0], rgba=[1, 0.7, 0, 0.7])
    for axis, rgba in enumerate(([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])):
        end = np.eye(3)[axis] * 0.12
        marker.add_site(
            name=f"target_axis_{axis}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, *end],
            size=[0.004, 0, 0],
            rgba=rgba,
        )
    # Compile now to catch scene/asset errors before launching the controller.
    spec.compile()
    path.write_text(spec.to_xml())


class ScanEnvironment(DefaultEnv):
    def check_fall(self):
        # Do not let the generic simulator silently teleport/reset a failed scan.
        if not self.elastic_band.enable:
            up = self.mj_data.body("pelvis").xmat.reshape(3, 3)[2, 2]
            if self.mj_data.qpos[2] < 0.45 or up < 0.5:
                raise RuntimeError("Robot fell or tilted excessively; scan aborted without resetting physics")


class ScanSimulation:
    def __init__(
        self,
        output,
        object_pose,
        radius,
        height,
        scanner_mass=0.15,
        headless=False,
        table_radius=0.45,
        table_height=0.72,
    ):
        scene_path = output / "scene.xml"
        make_scene(scene_path, object_pose, radius, height, scanner_mass, table_radius, table_height)
        config = SimLoopConfig(interface="sim", enable_onscreen=not headless).load_wbc_yaml()
        config.update(ROBOT_SCENE=str(scene_path), PRINT_SCENE_INFORMATION=False, USE_JOYSTICK=0)
        assert config["INTERFACE"] == "lo" and config["ENV_TYPE"] == "sim"
        ChannelFactoryInitialize(config["DOMAIN_ID"], "lo")
        self.env = ScanEnvironment(config, onscreen=not headless, offscreen=False)
        self.env.set_unitree_bridge(UnitreeSdk2Bridge(config))
        self.model, self.data = self.env.mj_model, self.env.mj_data
        self.dt = self.env.sim_dt
        self.object_pose = object_pose
        self.object_geom = self.model.geom("scan_object_surface").id
        self.obstacle_geoms = {self.object_geom, self.model.geom("scan_table_top").id}
        self.obstacle_geoms.update(self.model.geom(f"scan_table_leg_{i}").id for i in range(4))
        self.scanner_body = self.model.body("virtual_scanner").id
        self.target_mocap = self.model.body_mocapid[self.model.body("scan_target").id]
        self.scratch = mujoco.MjData(self.model)
        self.arm_joints = [
            self.model.joint(f"right_{n}_joint").id
            for n in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ]
        self.arm_qpos = self.model.jnt_qposadr[self.arm_joints]
        self.arm_limits = self.model.jnt_range[self.arm_joints].copy()
        if self.env.viewer:
            self.env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.env.viewer.cam.lookat[:] = object_pose.position / 2 + [0, 0, 0.65]
            self.env.viewer.cam.distance = 3.7
            self.env.viewer.cam.azimuth = 45
            self.env.viewer.cam.elevation = -25

    def body_pose(self, body, data=None):
        body = (self.data if data is None else data).body(body)
        return Pose(body.xpos, body.xquat)

    def body_speed(self, body):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.model.body(body).id, velocity, 0
        )
        return float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3]))

    def tracking_poses(self):
        return [self.body_pose(name) for name in TRACKING_BODIES]

    def set_target(self, world_pose):
        self.data.mocap_pos[self.target_mocap] = world_pose.position
        self.data.mocap_quat[self.target_mocap] = world_pose.quaternion

    def step(self):
        self.env.sim_step()
        # mj_step's derived poses refer to its pre-integration state. Refresh
        # kinematics before using feedback or recording a measurement.
        mujoco.mj_forward(self.model, self.data)
        for contact in self.data.contact:
            if self.obstacle_geoms.intersection((contact.geom1, contact.geom2)) and contact.dist < -0.002:
                raise RuntimeError("Robot/scanner contacted the scan object or table")

    def reachable(self, target_world, base_xy_world, base_yaw):
        """IK is a feasibility test on scratch data; it never actuates the robot."""
        d = self.scratch
        d.qpos[:] = self.data.qpos
        d.qpos[:3] = [*base_xy_world, 0.79]
        d.qpos[3:7] = quaternion(Rotation.from_euler("z", base_yaw))
        low, high = self.arm_limits[:, 0] + 0.025, self.arm_limits[:, 1] - 0.025
        initial = np.clip(d.qpos[self.arm_qpos], low, high)

        def residual(q):
            d.qpos[self.arm_qpos] = q
            mujoco.mj_forward(self.model, d)
            actual = self.body_pose(WRIST_BODY, d)
            rot_error = (rotation(target_world.quaternion) * rotation(actual.quaternion).inv()).as_rotvec()
            return np.r_[actual.position - target_world.position, 0.3 * rot_error]

        solution = least_squares(residual, initial, bounds=(low, high), max_nfev=80)
        residual(solution.x)
        pos_error, rot_error = pose_error(self.body_pose(WRIST_BODY, d), target_world)
        return pos_error < 0.02 and rot_error < np.deg2rad(6), pos_error, rot_error

    def capture(self, path, width=48, height=36, fov_degrees=48):
        """Capture actual scene intersections from the wrist-mounted scanner.

        The optical forward axis is +X; results contain world/object-frame point
        clouds and actual poses. Robot occlusions count: only first hits on the
        object's surface are retained. This is a virtual depth scan, not a
        physical scanner driver or a fabricated sample of the object mesh.
        """
        wrist = self.body_pose(WRIST_BODY)
        scanner = wrist.compose(SCANNER_IN_WRIST)
        rot = rotation(scanner.quaternion)
        extent = np.tan(np.deg2rad(fov_degrees / 2))
        points = []
        geom = np.zeros(1, dtype=np.int32)
        depth = np.full((height, width), np.nan, dtype=np.float32)
        for row, z in enumerate(np.linspace(extent * height / width, -extent * height / width, height)):
            for col, y in enumerate(np.linspace(extent, -extent, width)):
                direction = rot.apply(np.array([1, y, z]) / np.linalg.norm([1, y, z]))
                distance = mujoco.mj_ray(
                    self.model, self.data, scanner.position, direction, None, True, self.scanner_body, geom
                )
                if distance > 0 and geom[0] == self.object_geom:
                    points.append(scanner.position + distance * direction)
                    depth[row, col] = distance
        if len(points) < 20:
            raise RuntimeError(f"Scanner saw only {len(points)} object pixels; capture rejected")
        points = np.asarray(points)
        inverse = self.object_pose.inverse()
        object_points = rotation(inverse.quaternion).apply(points) + inverse.position
        np.savez_compressed(
            path,
            points_world=points,
            points_object=object_points,
            depth=depth,
            scanner_position=scanner.position,
            scanner_quaternion_wxyz=scanner.quaternion,
            wrist_position=wrist.position,
            wrist_quaternion_wxyz=wrist.quaternion,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            mocap_pos=self.data.mocap_pos.copy(),
            mocap_quat=self.data.mocap_quat.copy(),
            sim_time=float(self.data.time),
        )
        ply = path.with_suffix(".ply")
        with ply.open("w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            np.savetxt(f, object_points, fmt="%.6f")
        return {
            "path": str(path),
            "point_count": len(points),
            "actual_wrist_world": wrist.to_dict(),
            "actual_scanner_world": scanner.to_dict(),
        }

    def close(self):
        if self.env.viewer:
            self.env.viewer.close()
        bridge = self.env.unitree_bridge
        for item in vars(bridge).values():
            if hasattr(item, "Close"):
                item.Close()
