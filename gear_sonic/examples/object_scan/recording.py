"""Record measured simulation states and render them offline without changing control timing.

Record: python -m gear_sonic.examples.object_scan.run --headless --views 6 --record
Render: python -m gear_sonic.examples.object_scan.recording RUN_DIRECTORY --ffmpeg /path/to/ffmpeg
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from gear_sonic.examples.object_scan.planning import pose_error


class StateRecording:
    def __init__(self):
        self.frames = []

    def append(self, sim, session):
        if session.phase == "initializing":
            return
        error = (np.nan, np.nan)
        if session.goal is not None:
            error = pose_error(sim.body_pose("right_wrist_yaw_link"), session.wrist_target)
        self.frames.append(
            {
                "time": float(sim.data.time),
                "wall_elapsed": time.monotonic() - session.started,
                "qpos": sim.data.qpos.copy(),
                "qvel": sim.data.qvel.copy(),
                "mocap_pos": sim.data.mocap_pos.copy(),
                "mocap_quat": sim.data.mocap_quat.copy(),
                "phase": session.phase,
                "view": session.goal.index if session.goal else 0,
                "captures": len(session.scans),
                "position_error_m": error[0],
                "orientation_error_degrees": np.rad2deg(error[1]),
            }
        )

    def save(self, output):
        if not self.frames:
            return None
        path = output / "recording.npz"
        np.savez_compressed(path, **{key: np.asarray([f[key] for f in self.frames]) for key in self.frames[0]})
        return str(path)


def render_video(directory, ffmpeg, fps=25, width=1280, height=720):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import mujoco

    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    recording = np.load(directory / "recording.npz", allow_pickle=False)
    times = recording["time"]
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Recording needs monotonically increasing simulation timestamps")
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [config["object_distance"] * 0.65, 0, 0.66]
    camera.distance, camera.azimuth, camera.elevation = 3.6, 45, -23
    video_path = directory / "scan_video.mp4"
    frames = int(np.ceil((times[-1] - times[0]) * fps)) + 1
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    banner_start = -1.0
    last_capture = 0
    snapshot_views = set()
    with (directory / "video_encode.log").open("w") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            with mujoco.Renderer(model, height=height, width=width) as renderer:
                for frame in range(frames):
                    sample_time = min(times[0] + frame / fps, times[-1])
                    # Use a recorded state, never interpolate an invented motion.
                    i = max(0, int(np.searchsorted(times, sample_time, side="right") - 1))
                    for key in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
                        getattr(data, key)[:] = recording[key][i]
                    data.time = times[i]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    rgb = renderer.render().copy()
                    cv2.rectangle(rgb, (0, 0), (width, 91), (15, 24, 38), -1)
                    cv2.putText(
                        rgb,
                        "SONIC 1.1 | G1 object scanning",
                        (26, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (242, 245, 250),
                        2,
                        cv2.LINE_AA,
                    )
                    captures = int(recording["captures"][i])
                    phase = str(recording["phase"][i])
                    view = int(recording["view"][i])
                    label = (
                        f"View {view}/{config['views']}   {phase.upper()}   Captured {captures}/{config['views']}"
                    )
                    cv2.putText(
                        rgb, label, (26, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (120, 215, 220), 1, cv2.LINE_AA
                    )
                    cv2.rectangle(rgb, (0, height - 66), (width, height), (15, 24, 38), -1)
                    elapsed = times[i] - times[0]
                    text = f"Recorded simulation | 1x sim speed | t={elapsed:05.1f}s"
                    cv2.putText(
                        rgb,
                        text,
                        (26, height - 39),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (220, 228, 237),
                        1,
                        cv2.LINE_AA,
                    )
                    if np.isfinite(recording["position_error_m"][i]):
                        error = (
                            f"Measured wrist error: {recording['position_error_m'][i] * 100:.2f} cm / "
                            f"{recording['orientation_error_degrees'][i]:.2f} deg"
                        )
                        cv2.putText(
                            rgb,
                            error,
                            (26, height - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (220, 228, 237),
                            1,
                            cv2.LINE_AA,
                        )
                    if captures > last_capture:
                        banner_start = elapsed
                        last_capture = captures
                    if banner_start >= 0 and elapsed - banner_start < 1.2:
                        cv2.rectangle(rgb, (width - 340, 109), (width - 24, 154), (24, 105, 79), -1)
                        cv2.putText(
                            rgb,
                            f"SCAN {captures} CAPTURED",
                            (width - 320, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.72,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    encoder.stdin.write(rgb.tobytes())
                    if captures and captures not in snapshot_views:
                        cv2.imwrite(
                            str(directory / f"video_scan_{captures:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        )
                        snapshot_views.add(captures)
                    if frame % (fps * 10) == 0:
                        print(f"Rendered {frame}/{frames} frames", flush=True)
                # Explicitly labelled end card; no extra simulated movement.
                cv2.rectangle(rgb, (width - 460, 109), (width - 24, 154), (24, 105, 79), -1)
                cv2.putText(
                    rgb,
                    f"{summary['status'].upper()}: {captures}/{config['views']} scans",
                    (width - 440, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                for _ in range(2 * fps):
                    encoder.stdin.write(rgb.tobytes())
        finally:
            encoder.stdin.close()
            returncode = encoder.wait()
        if returncode:
            raise RuntimeError(f"Video encoder failed ({returncode}); inspect video_encode.log")
    metadata = {
        "path": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames + 2 * fps,
        "simulation_start": float(times[0]),
        "simulation_end": float(times[-1]),
        "end_card_seconds": 2,
        "method": (
            "Offline rendering of measured MuJoCo states; startup initialization omitted; 1x simulation speed"
        ),
    }
    (directory / "video_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(video_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("ffmpeg was not found; provide --ffmpeg /path/to/ffmpeg")
    render_video(args.directory, args.ffmpeg)
