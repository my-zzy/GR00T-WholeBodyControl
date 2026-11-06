Adaptive object scanning with SONIC v1.1 (MuJoCo only)
====================================================

Start from the repository root, after the existing SONIC setup:

  conda activate sonic
  python -m gear_sonic.examples.object_scan.run --views 3

This ONE command starts the MuJoCo viewer, a physical G1 simulation, the
pretrained SONIC v1.1 C++ controller, a Python goal controller, and a virtual
scanner. Close the viewer or press Ctrl+C to stop. It automatically starts
the policy and releases the startup support band. Do not also run
run_sim_loop.py or deploy.sh: the launcher owns both processes.

For a bounded run without a desktop:

  python -m gear_sonic.examples.object_scan.run --headless --views 3

The first launch builds cached TensorRT engines for v1.1 if necessary.
There is no model training, no new dependency installation, and no system
configuration change. The existing sonic environment must be active.
The launcher forces DDS to loopback and has no real-robot mode.

Recording a video
-----------------

  python -m gear_sonic.examples.object_scan.run --headless --views 6 --record
  python -m gear_sonic.examples.object_scan.recording RUN_DIRECTORY --ffmpeg /path/to/ffmpeg

The first command saves actual MuJoCo states in recording.npz at the control
ticks, starting when policy control is released. The second renders a 720p,
25 fps H.264 scan_video.mp4 offline, so rendering cannot slow the controller.
Playback is at 1x simulation speed with phase/error overlays and a labelled
2-second final hold. Controller initialization is omitted. No synthetic
motion interpolation is used. ffmpeg must include the libx264 encoder.

On this machine an existing compatible binary is available at:
  /home/user/miniconda3/envs/ambench/bin/ffmpeg

video_metadata.json records the time mapping, resolution and frame count.
video_scan_001.png, etc. are stills at each capture. Offline rendering needs
MuJoCo EGL/OpenGL access, just like the simulation viewer.

What happens
------------

1. Load a small cylinder centred on a round table and the G1 model with a
   wrist-mounted virtual scanner. Defaults: object diameter 0.36 m, height
   0.30 m; table diameter 0.90 m, tabletop 0.72 m above ground.
2. Generate one object-relative right-wrist goal. Targets follow a perturbed
   circular arc, with independently varied height, radius, pitch, and roll.
   Wrist pitch aims the scanner toward the object's centre height.
3. Choose a base stance behind/left of that right-wrist target. Check arm
   reachability with joint-limited IK on separate scratch MuJoCo data.
   IK NEVER sets the real simulation's joint positions or motor commands.
4. Walk toward the stance while facing the object. Arc/radial feedback avoids
   cutting through the table/object footprint. All motor commands come from SONIC.
5. Stand, smoothly move the right wrist to the goal, and apply bounded pose
   bias correction. Head and left wrist keep comfortable body-relative targets.
6. Require the MEASURED wrist position, orientation, wrist speed, and base
   speed to satisfy the arrival gate continuously before capturing.
7. Raycast from the ACTUAL scanner pose into MuJoCo geometry, saving a point
   cloud and metadata. Only then ask the view planner for its next goal.
8. Retract the arm, move to the next view, and repeat.

The base/wrist feedback comes from actual MuJoCo kinematics. The controller
does not use the upstream base_trans_measured visualization placeholder.
Falls, object/table contacts, unreachable goals, controller exits, and timeouts
abort the run and produce a failed summary. They never count as a scan and
never silently reset or teleport the robot.

Defaults and useful options
---------------------------

  --views 3                       Number of successful scans requested
  --seed 7                        Reproducible target sampling
  --height-min 0.82 --height-max 1.04
                                  Right-wrist heights, metres above ground
  --wrist-radius 0.95             Approximate wrist distance from object centre
  --step-min-degrees 8 --step-max-degrees 16
                                  Random angular increment after each scan
  --object-radius 0.18 --object-height 0.30
                                  Cylinder dimensions in metres
  --table-radius 0.45 --table-height 0.72
                                  Round table radius and tabletop height
  --scanner-mass 0.15             Simulated wrist payload, kilograms
  --speed 0.30                    Commanded slow-walk speed, m/s ([0.2, 0.4])
  --position-tolerance 0.05       Measured wrist error limit, metres
  --orientation-tolerance-degrees 10
                                  Measured wrist rotation error limit
  --hold-seconds 0.75             Continuous settled interval before capture
  --view-timeout 150              Maximum wall time per goal, seconds
  --output /absolute/new/folder   New run directory; must not already exist

Use --help for all options. These are prototype tolerances, not an accuracy
guarantee for SONIC or for a physical scanner. Tightening them can correctly
produce a timeout. A kinematically reachable target does not guarantee the
learned policy will track it within a particular tolerance.

Artifacts
---------

By default each run is stored in .cache/object_scan/<timestamp>/:

  config.json            Run parameters
  scene.xml              Exact generated scene, including scanner mass
  controller.log         Native SONIC initialization/inference output
  telemetry.jsonl        Measured base/wrist states and pose errors at 50 Hz
  goal_001.json          The accepted target, in the object frame
  scan_001.npz           Actual depth, world/object point clouds, physics state
  scan_001.ply           Object-frame point cloud, readable by common viewers
  scan_001.json          Target, actual poses, errors, and capture metadata
  summary.json           complete/failed/interrupted and all accepted scans

The virtual scanner uses a 48 x 36 ray grid with a 48-degree horizontal field
of view. Only first intersections on the object's surface are saved; robot
and table occlusions are respected. Table pixels are excluded from the point
cloud. Captures with fewer than 20 object pixels fail.
It is an ideal geometry-based depth scanner, without noise, exposure timing,
reflectance, or a physical-device driver. Its housing contributes inertia;
housing collision is disabled to avoid interference with the stock hand mesh.

Frames and integration with your own view planner
------------------------------------------------

Object coordinates: origin on the ground directly below the cylinder/table
centre; metres; +Z up. Wrist target heights remain relative to the ground.
The cylinder sits from Z = table_height to table_height + object_height.
The object frame is translated along world +X by --object-distance
(default 1.35 m). Quaternions are always [w, x, y, z]. A goal targets the
right_wrist_yaw_link frame. Its +X axis points approximately toward the object.

The virtual scanner's optical frame is offset [0.22, 0.025, 0] metres from
the right wrist, with identical orientation and optical forward +X. This
SCANNER_IN_WRIST transform in simulation.py is separate from SONIC's virtual
three-point tracking offsets. The code applies BOTH transforms where needed.
If a real view planner supplies scanner-frame goals, convert them first:

  wrist_goal = scanner_goal.compose(SCANNER_IN_WRIST.inverse())

The right wrist, left wrist, and head/torso references are converted to the
pelvis frame on every control tick, including SONIC's tracking-point offsets.
The actual wrist goal stays fixed in the object/world frame while standing.

To replace random views, provide a Python module with a factory:

  def make_planner(config):
      return MyPlanner()

  class MyPlanner:
      def next_view(self, previous_scan):
          # previous_scan is None for the first call. Later calls receive
          # the actual capture paths, measured poses, and tracking errors.
          # Return a ScanGoal with a unique positive index and a Pose in
          # the object coordinate frame.
          return ScanGoal(index, Pose(position_xyz, quaternion_wxyz))

  python -m gear_sonic.examples.object_scan.run --planner my_module:make_planner

Import Pose and ScanGoal from gear_sonic.examples.object_scan.planning.
External goals are also checked for clearance and reachability. Unlike the
random generator, they are not silently replaced if unreachable. The hook
runs in a background thread while G1 holds its carrying pose and the
simulation/control loop continues. It may wait for an external program's
response; --view-timeout also bounds this wait. Planner exceptions abort the
run. Do not access MuJoCo or control sockets from the planner thread. For
heavy CPU-bound planning, put that computation in a separate process/service.

Scope
-----

This is a simulation prototype for a single cylindrical object on a round
table, on flat ground. It has geometric table/object clearance, not a general
obstacle-map navigator.
Physical deployment would additionally need object-relative localization,
scanner calibration/driver, measured payload parameters, and validation of
the required pose accuracy and settling behaviour.

Tests
-----

  python -m unittest gear_sonic.examples.object_scan.test_object_scan -v

The tests cover object/root transforms and wrist tracker offsets, shortest-path
rotation interpolation, arc navigation/clearance, sequential goal generation,
the measured arrival gate, input validation, and actual MuJoCo ray intersections.
The headless command above additionally exercises the pretrained controller.
