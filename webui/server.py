#!/usr/bin/env python3
"""Web interface for the arm pipelines: camera + landmarks, record, coach, mirror.

    webui/run.sh                                        # then open http://localhost:8765
    webui/run.sh --camera http://<phone-ip>:4747/video
    webui/run.sh --demo-dir ~/demos                     # also offer demos from another folder

The server owns the one camera connection (DroidCam allows a single viewer) and
hands frames to whichever mode is running. It starts and stops the ROS launch
files itself, so run it through run.sh (ROS sourced, MediaPipe venv). The
simulated robot (MoveIt, controllers, RViz) starts with the server and stays up
across modes; Record and Mirror only start their IK node (--no-robot: start the
robot when a mode first needs it). Recordings, coaching logs and demos made from recordings go to
the git-ignored local/.
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "MediaPipe"))

import arm_xy  # noqa: E402
from run_arm_xy import draw_skeleton, save_trajectory  # noqa: E402
from therapy.coach import GOOD, Coach, FakeChild, Voice  # noqa: E402
from therapy.demo import Demo  # noqa: E402
from therapy.observer import Observer, draw_sample  # noqa: E402
from therapy.robot_link import Robot, SocketRobot, make_robot  # noqa: E402
from therapy.retarget import ARMS  # noqa: E402
from therapy.robot_path import RobotKinematics  # noqa: E402

PAGE = Path(__file__).resolve().parent / "index.html"
LOCAL = REPO_ROOT / "local"                 # git-ignored: data written on this machine
RECORDINGS = LOCAL / "recordings"
SESSIONS = LOCAL / "sessions"
DEMO_CACHE = LOCAL / "demos"                # robot demos made from recordings (recording_demo.py)
DEFAULT_DEMO_SOURCES = [                    # (folder, pattern) the Coach tab lists; --demo-dir adds more
    (RECORDINGS, "*.json"),
    (REPO_ROOT / "therapy" / "demos", "*.csv"),
    (REPO_ROOT, "*.csv"),
]
MAX_UNREACHABLE = 0.5                       # more of a recording's poses than this out of reach -> don't coach with it
# RViz layout without MoveIt's goal-state robot, which would sit still next to the moving arm.
RVIZ_CONFIG = "config/mediapipe_replay.rviz"
ROBOT_PROCESSES = ("robot", "play_demo")    # the sim robot, kept running across modes
MODE_PROCESSES = ("mirror",)                # started and stopped by Record / Mirror
PLAY_DEMO = REPO_ROOT / "bracketbot_moveit_config" / "scripts" / "play_demo.py"
HOME_SECONDS = 2.0                         # glide back to the zero pose after mirroring

PHASE_TITLES = {
    "waiting": "Waiting", "recording": "Recording your movement", "starting": "Starting the robot",
    "preparing": "Preparing the demonstration",
    "calibrating": "Measuring your arm", "calibrated": "Arm measured",
    "getting_ready": "Robot getting ready", "demonstrating": "Robot is demonstrating",
    "returning": "Robot returning to its start pose", "your_turn": "Your turn",
    "adapting": "Adjusting the next demonstration", "resting": "Short rest",
    "finished": "Session finished", "stopped": "Stopped", "mirroring": "Robot is mirroring you", "error": "Problem",
}

WHY = {
    "reach": "the hand didn't reach as far as the robot's, so the next demo is smaller and slower",
    "height": "the hand didn't get as high, so the robot will pause at the top of the movement",
    "elbow": "the elbow didn't straighten as much, so the robot shows just the stretch, slower",
    "smoothness": "it wasn't one smooth movement, so the robot slows down",
    "tempo": "the pace was different, so the robot changes its speed",
    "direction": "the hand went a different way, so the robot slows down",
    "shape": "the hand's path looked different, so the robot slows down",
}


class ApiError(Exception):
    pass


def placeholder(text, height=480, width=640):
    frame = np.full((height, width, 3), 32, np.uint8)
    cv2.putText(frame, text[:48], (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return frame


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def explain(info):
    """One sentence on why the coach changed (or kept) the demonstration."""
    if not info["moved"]:
        return "No movement was seen, so the robot repeats the same demo."
    if info["message"].startswith("Let's take a short break"):
        return "Scores dropped three reps in a row, so there's a short rest and a smaller demo."
    if info["good"]:
        if info["plan_after"] != info["plan_before"]:
            return "Three good reps in a row, so the demo moves back toward the full movement."
        return "Good rep, so the robot keeps the same demo."
    return f"Score {info['score']:.0f}: {WHY.get(info['weakest'], 'the robot adjusts the demo')}."


def display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class View:
    """Latest annotated frame as JPEG, for the MJPEG stream."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg, self._seq = None, 0

    def publish(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._cond:
                self._jpeg, self._seq = buf.tobytes(), self._seq + 1
                self._cond.notify_all()

    def wait(self, seq, timeout=5.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seq, timeout=timeout)
            return self._seq, self._jpeg


class Camera:
    """The single camera connection; every mode reads frames from here."""

    def __init__(self, spec):
        self.spec = str(spec)
        self.error = None
        self._cap = None
        self._frame, self._seq = None, 0
        self._cond = threading.Condition()
        self._reopen = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def fake(self):
        return self.spec == "fake"

    def switch(self, spec):
        self.spec = str(spec)
        self._reopen.set()

    def _run(self):
        while True:
            if self._reopen.is_set():
                self._reopen.clear()
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
            if self.fake:
                frame = placeholder("no camera (simulated child)")
                self.error = None
                time.sleep(1 / 15)
            else:
                if self._cap is None:
                    try:
                        self._cap = arm_xy.open_camera(self.spec, 640, 480, fps=30)
                        self.error = None
                    except SystemExit as e:
                        self.error = str(e)
                        time.sleep(2.0)
                        continue
                ok, frame = self._cap.read()
                if not ok:
                    self.error = f"no frames from camera {self.spec}"
                    time.sleep(0.05)
                    continue
                self.error = None
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def next_frame(self, after_seq, timeout=5.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != after_seq, timeout=timeout)
            return self._seq, self._frame


class FrameReader:
    """cv2.VideoCapture-like view of the shared Camera, handed to therapy.Observer."""

    def __init__(self, camera):
        self.camera = camera
        self._seq = camera.next_frame(-1, timeout=0)[0]

    def read(self):
        seq, frame = self.camera.next_frame(self._seq)
        if frame is None or seq == self._seq:
            return False, None
        self._seq = seq
        return True, frame.copy()

    def isOpened(self):
        return True

    def get(self, prop):
        return 0.0

    def release(self):
        pass


def _proc_stat(pid):
    """(process group, start time in clock ticks) of a process, or None if it's gone."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return int(fields[2]), int(fields[19])


def _group_members(pgid, since):
    """Pids in process group pgid that started no earlier than `since` (so not a later reuse of the id)."""
    members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            stat = _proc_stat(entry.name)
            if stat and stat[0] == pgid and stat[1] >= since:
                members.append(int(entry.name))
    return members


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Processes:
    """ROS commands started from the page. All of them are stopped together."""

    def __init__(self, on_line, registry=None):
        self._procs = []
        self._on_line = on_line
        self._lock = threading.Lock()
        # Process groups written here, so the next server can stop what a killed one left running.
        self._registry = registry

    def _save_registry(self):
        """Call with the lock held."""
        if self._registry is None:
            return
        entries = []
        for name, proc in self._procs:
            stat = _proc_stat(proc.pid)
            if stat is not None:
                entries.append({"name": name, "pgid": proc.pid, "start": stat[1]})
        self._registry.parent.mkdir(parents=True, exist_ok=True)
        self._registry.write_text(json.dumps(entries))

    def reap_orphans(self):
        """Stop the ROS commands a previous server left running (killed, or its terminal closed).
        Returns their names."""
        try:
            entries = json.loads(self._registry.read_text())
        except (OSError, ValueError, AttributeError):
            return []
        stale = [e for e in entries if _group_members(e["pgid"], e["start"])]
        for sig, grace in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 1.0)):
            alive = [e for e in stale if _group_members(e["pgid"], e["start"])]
            if not alive:
                break
            for e in alive:
                try:
                    os.killpg(e["pgid"], sig)
                except (ProcessLookupError, PermissionError):
                    pass
            deadline = time.time() + grace
            while time.time() < deadline and any(_group_members(e["pgid"], e["start"]) for e in alive):
                time.sleep(0.2)
        self._registry.unlink(missing_ok=True)
        return [e["name"] for e in stale]

    @staticmethod
    def _env():
        """This process's environment minus what importing OpenCV added. Its bundled Qt plugins
        make RViz abort with 'Could not load the Qt platform plugin "xcb"'."""
        env = {k: v for k, v in os.environ.items() if not (k.startswith("QT_") and "/cv2/" in v)}
        libs = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and "/cv2/" not in p]
        if libs:
            env["LD_LIBRARY_PATH"] = ":".join(libs)
        else:
            env.pop("LD_LIBRARY_PATH", None)
        return env

    def start(self, name, cmd):
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=self._env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        with self._lock:
            self._procs.append((name, proc))
            self._save_registry()
        threading.Thread(target=self._pump, args=(name, proc), daemon=True).start()
        return proc

    def _pump(self, name, proc):
        for line in proc.stdout:
            self._on_line(name, line.rstrip())

    def running(self):
        with self._lock:
            return sorted({name for name, proc in self._procs if proc.poll() is None})

    def stop_all(self):
        self.stop()

    def stop(self, names=None):
        """Stop the commands with these names (default: all of them)."""
        with self._lock:
            procs = [(n, p) for n, p in self._procs if names is None or n in names]
            self._procs = [(n, p) for n, p in self._procs if not (names is None or n in names)]
            self._save_registry()
        # SIGINT to the launch process only: it shuts its own nodes down in order. Each command runs in
        # its own session, so its process group also catches nodes left behind if the launch exits first.
        for _, proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        deadline = time.time() + 20.0
        for _, proc in procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                pass
        for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 1.0)):
            alive = [proc.pid for _, proc in procs if _group_alive(proc.pid)]
            if not alive:
                break
            for pgid in alive:
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    pass
            time.sleep(grace)
        for _, proc in procs:
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass


class ReportingRobot(Robot):
    """Passes demonstrations through. A robot that can't play one ends the session with the reason
    on the page, rather than coaching against a robot that never moved."""

    def __init__(self, inner, app):
        self.inner, self.app = inner, app

    def play(self, demo, on_event=None):
        try:
            return self.inner.play(demo, on_event)
        except Exception as e:
            self.app.robot_error = str(e)
            self.app.stop.set()
            return 0.0

    def close(self):
        self.inner.close()


class WebCoach(Coach):
    """Coach whose window is the web page: frames go to the stream, Stop ends the session."""

    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = app

    def show(self, frame, sample, msg):
        if self.app.stop.is_set():
            raise KeyboardInterrupt
        self.app.live = msg
        if frame is None:
            frame = placeholder(msg or "")
        elif hasattr(self.obs, "arm"):
            frame = draw_sample(frame, sample, self.obs.arm)
        self.app.view.publish(frame)


class App:
    def __init__(self, camera_spec, demo_dirs=(), start_robot=True, registry=None):
        """registry: file listing the ROS commands this server runs (one per port, so servers on other
        ports are left alone); what a killed server left running is stopped on start."""
        self.camera = Camera(camera_spec)
        self.view = View()
        self.demo_sources = DEFAULT_DEMO_SOURCES + [(Path(d).expanduser().resolve(), pattern)
                                                    for d in demo_dirs for pattern in ("*.json", "*.csv")]
        self.mode = "idle"                  # idle | record | coach | mirror
        self.phase, self.detail, self.live = "waiting", "Starting the camera...", ""
        self.phase_info, self.phase_since = {}, time.time()
        self.tracked = False
        self.side, self.plane = "right", "side"
        self.recording, self.recording_t, self.record_mirrored = [], [], False
        self.reps = []
        self.log = deque(maxlen=80)
        self.ik = ""
        self.stop = threading.Event()
        self.robot_error = None
        self.sim_port = None                # play_demo.py --serve port of the sim this server started
        self.publishing = False             # landmarks go to the live IK launch (mirror, or record with robot)
        self.camera_owner = "preview"       # which thread reads frames: preview loop or coach
        self.procs = Processes(self._on_process_line, registry=registry)
        self._stopper = None                # background thread stopping the ROS processes, if any
        self._landmark_pub = None
        self._home_pubs = {}
        self._ros_node = None
        self.robot_state = "off"            # off | starting | ready | error
        self._robot_lock = threading.Lock()
        self._closing = threading.Event()
        left = self.procs.reap_orphans()
        if left:
            message = f"Stopped ROS processes a previous server left running: {', '.join(left)}"
            self.log.append(f"{time.strftime('%H:%M:%S')}  {message}")
            print(message, flush=True)
        threading.Thread(target=self._preview_loop, daemon=True).start()
        if start_robot and shutil.which("ros2") is not None:
            self._start_robot_background()

    # ---- status -----------------------------------------------------------

    def set_status(self, phase, detail, info=None):
        if (phase, detail) != (self.phase, self.detail):
            self.phase, self.detail = phase, detail
            self.phase_info, self.phase_since = info or {}, time.time()
            self.log.append(f"{time.strftime('%H:%M:%S')}  {PHASE_TITLES.get(phase, phase)}: {detail}")

    def demos(self):
        """Demonstrations the Coach tab offers: recordings (turned into robot demos with IK when used) first,
        newest first, then robot demo CSVs, newest first."""
        recordings, csvs, seen = [], [], set()
        for folder, pattern in self.demo_sources:
            for path in folder.glob(pattern):
                if path in seen or path.parent == DEMO_CACHE:
                    continue
                seen.add(path)
                if path.suffix == ".json":
                    recordings.append((path.stem.rsplit("_", 1)[-1], path))
                else:
                    csvs.append((path.stat().st_mtime, path))
        items = [{"path": display_path(p), "label": f"recording: {p.stem}", "kind": "recording"}
                 for _, p in sorted(recordings, reverse=True)]
        items += [{"path": display_path(p), "label": f"robot demo: {display_path(p)}", "kind": "robot demo"}
                  for _, p in sorted(csvs, reverse=True)]
        return items

    def snapshot(self):
        return {
            "now": time.time(),
            "mode": self.mode,
            "phase": self.phase,
            "title": PHASE_TITLES.get(self.phase, self.phase),
            "detail": self.detail,
            "phase_since": self.phase_since,
            "phase_info": {k: v for k, v in self.phase_info.items() if k in ("rep", "seconds", "corrected", "plan")},
            "live": self.live if self.mode == "coach" else "",
            "tracked": self.tracked,
            "recording_frames": len(self.recording) if self.mode == "record" else 0,
            "camera": {"spec": self.camera.spec, "error": self.camera.error},
            "ik": self.ik,
            "reps": self.reps[-20:],
            "log": list(self.log)[-40:],
            "ros": shutil.which("ros2") is not None,
            "robot": self.robot_state,
            "processes": self.procs.running(),
            "demos": self.demos(),
            # newest first by the timestamp in the name; the name starts with the arm, so a plain sort misorders
            "recordings": [p.name for p in sorted(RECORDINGS.glob("*.json"),
                                                   key=lambda p: p.stem.rsplit("_", 1)[-1], reverse=True)][:8],
        }

    def _on_process_line(self, name, line):
        if line.find("IK:") != -1 and "solved" in line:
            self.ik = line[line.find("IK:"):]
        elif "Traceback" in line or "process has died" in line or "[ERROR]" in line and "octomap" not in line:
            self.log.append(f"{time.strftime('%H:%M:%S')}  [{name}] {line[-200:]}")

    def _require_idle(self):
        if self.mode != "idle":
            raise ApiError(f"Stop {self.mode} first.")

    def _require_ros(self):
        if shutil.which("ros2") is None:
            raise ApiError("ROS isn't available: start the server with webui/run.sh.")

    def _ensure_ros_node(self):
        if self._ros_node is None:
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            self._ros_node = rclpy.create_node("webui")
        return self._ros_node

    def _stop_mode_processes(self):
        self.publishing = False
        if self._stopper is not None and self._stopper is not threading.current_thread():
            self._stopper.join()
        self.procs.stop(MODE_PROCESSES)

    def _stop_mode_processes_async(self):
        """Stop Record / Mirror's IK node, then glide the arm back to its zero pose. The robot stays up."""
        self.publishing = False
        side = self.side

        def stop():
            self.procs.stop(MODE_PROCESSES)
            self._send_home(side)

        stopper = threading.Thread(target=stop, daemon=True)
        self._stopper = stopper
        stopper.start()

    def _send_home(self, side):
        pub = self._home_pubs.get(side)
        if pub is None or self.robot_state != "ready":
            return
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        prefix = ARMS[side]["prefix"]
        point = JointTrajectoryPoint(positions=[0.0] * 7, time_from_start=Duration(sec=int(HOME_SECONDS)))
        pub.publish(JointTrajectory(joint_names=[f"{prefix}{i}" for i in range(7)], points=[point]))

    # ---- the simulated robot, shared by every mode --------------------------

    def _robot_up(self):
        return bool(self.sim_port and all(name in self.procs.running() for name in ROBOT_PROCESSES)
                    and SocketRobot("127.0.0.1", self.sim_port).ping())

    def ensure_robot(self, cancel=None, announce=False):
        """Start the simulated robot unless this server's one is already up. Blocks until it answers."""
        with self._robot_lock:
            if self._robot_up():
                self.robot_state = "ready"
                return self.sim_port
            try:
                self._start_sim(cancel or self._closing, announce)
            except Exception as e:
                self.robot_state = "error"
                self.log.append(f"{time.strftime('%H:%M:%S')}  Robot didn't start: {e}")
                raise
            return self.sim_port

    def _start_robot_background(self):
        if self.robot_state == "starting":
            return

        def run():
            try:
                self.ensure_robot()
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()

    # ---- camera preview, recording and mirroring --------------------------

    def _preview_loop(self):
        seq = -1
        while True:
            if self.camera_owner != "preview":
                time.sleep(0.05)
                continue
            new_seq, frame = self.camera.next_frame(seq, timeout=1.0)
            if frame is None or new_seq == seq:
                if self.camera.error:
                    self.view.publish(placeholder(self.camera.error))
                    if self.mode == "idle" and self.phase == "waiting":
                        self.detail = f"Camera problem: {self.camera.error}"
                continue
            seq = new_seq
            if self.camera.fake:
                self.view.publish(frame)
                self._update_detail()
                continue
            pts = arm_xy.arm_xy(frame, side=self.side, plane=self.plane)
            self.tracked = pts is not None
            if pts is not None:
                if self.mode == "record":
                    self.recording.append(pts)
                    self.recording_t.append(time.monotonic())
                if self.publishing:
                    self._publish_landmarks(pts)
            shown = frame.copy()
            draw_skeleton(shown)
            self.view.publish(cv2.flip(shown, 1))
            self._update_detail()

    def _robot_detail(self):
        if not self.publishing:
            return ""
        follow = "The robot is following your arm." if self.tracked else \
            "Can't see your arm, so the robot holds its pose."
        if self.robot_state != "ready":
            return f" {follow} The robot is still starting in RViz (about 20 s)."
        return f" {follow} {self.ik or 'Connecting to MoveIt...'}"

    def _update_detail(self):
        if self.mode == "record":
            self.phase = "recording"
            self.detail = (f"Recording your {self.side} arm: {len(self.recording)} frames. Press Stop to save."
                           if self.tracked else "Recording paused: step back so your hips, shoulders and arm are visible."
                           ) + self._robot_detail()
        elif self.mode == "mirror":
            self.phase = "mirroring"
            self.detail = self._robot_detail().strip()
        elif self.mode == "idle" and self.phase == "waiting":
            self.detail = ("Camera ready. Choose Record, Coach or Mirror." if self.tracked or self.camera.fake
                           else "Camera ready. Stand so your hips, shoulders and arm are in view.")

    def set_camera(self, spec):
        self._require_idle()
        self.camera.switch(spec.strip())
        self.set_status("waiting", f"Switching camera to {spec.strip()}...")

    def _start_live_ik(self, side, plane, mirrored):
        """Start the live IK node, which drives the arm through its trajectory controller; the preview loop
        then publishes the landmarks to it. The robot is started too if it isn't up (without waiting)."""
        self._require_ros()
        self._stop_mode_processes()
        node = self._ensure_ros_node()
        if self._landmark_pub is None:
            from geometry_msgs.msg import PoseArray
            from trajectory_msgs.msg import JointTrajectory
            self._landmark_pub = node.create_publisher(PoseArray, "/mediapipe/arm_landmarks", 10)
            # made now so they're connected by the time a mode stops and sends the arm home
            self._home_pubs = {s: node.create_publisher(JointTrajectory, f"/{s}_arm_controller/joint_trajectory", 10)
                               for s in ARMS}
        self.side, self.plane, self.ik = side, plane, ""
        self.procs.start("mirror", ["python3", str(REPO_ROOT / "main_controller.py"), "--ros-args",
                                    "-p", "source:=live", "-p", "output:=controller", "-p", f"side:={side}",
                                    "-p", f"plane:={plane}", "-p", f"mirrored:={str(bool(mirrored)).lower()}"])
        self.publishing = True
        if not self._robot_lock.locked():
            self._start_robot_background()

    def _publish_landmarks(self, pts):
        from geometry_msgs.msg import Point, Pose, PoseArray
        msg = PoseArray()
        msg.header.stamp = self._ros_node.get_clock().now().to_msg()
        msg.header.frame_id = f"mediapipe_{self.plane}"
        msg.poses = [Pose(position=Point(x=float(x), y=float(y))) for x, y in pts]
        self._landmark_pub.publish(msg)

    def record_start(self, side="right", plane="side", show_robot=True, mirrored=False):
        self._require_idle()
        if self.camera.fake:
            raise ApiError("Recording needs a real camera.")
        self.side, self.plane = side, plane
        self.recording, self.recording_t, self.record_mirrored = [], [], bool(mirrored)
        if show_robot and shutil.which("ros2") is not None:
            self._start_live_ik(side, plane, mirrored)
        elif show_robot:
            self.log.append(f"{time.strftime('%H:%M:%S')}  ROS isn't sourced, so recording without the robot.")
        self.mode = "record"

    def record_stop(self):
        if self.mode != "record":
            raise ApiError("Not recording.")
        frames, times = self.recording, self.recording_t
        self.recording, self.recording_t = [], []
        self.mode = "idle"
        if self.publishing:
            self._stop_mode_processes_async()
        if len(frames) < 2:
            self.set_status("waiting", "Nothing was recorded: your arm wasn't visible.")
            return None
        take = SimpleNamespace(side=self.side, plane=self.plane, mirrored=self.record_mirrored)
        _, path, _ = save_trajectory(frames, take, out_dir=RECORDINGS, t=times)
        self.set_status("waiting", f"Saved {len(frames)} frames to local/recordings/{path.name}. "
                                   "It's now the first demo in the Coach tab.")
        return path.name

    def mirror_start(self, side="right", plane="side", mirrored=False):
        self._require_idle()
        if self.camera.fake:
            raise ApiError("Mirroring needs a real camera.")
        self._start_live_ik(side, plane, mirrored)
        self.mode = "mirror"
        self.log.append(f"{time.strftime('%H:%M:%S')}  Mirror started ({side} arm, {plane} view)")

    def mirror_stop(self):
        if self.mode != "mirror":
            raise ApiError("Not mirroring.")
        self.mode = "idle"
        self.set_status("waiting", "Mirroring stopped.")
        self._stop_mode_processes_async()

    # ---- coaching ---------------------------------------------------------

    def coach_start(self, demo, robot="stub", reps=8, arm=None, speak=True):
        self._require_idle()
        if demo not in {item["path"] for item in self.demos()}:
            raise ApiError(f"Unknown demo {demo!r}: pick one from the list.")
        if demo.endswith(".json") and robot != "sim":
            raise ApiError("A recording is turned into a robot demo with the simulated robot's IK: choose "
                           "'simulated robot'.")
        self.mode = "coach"
        self.reps = []
        self.robot_error = None
        self.stop.clear()
        threading.Thread(target=self._coach_thread, args=(demo, robot, int(reps), arm or None, bool(speak)),
                         daemon=True).start()

    def coach_stop(self):
        if self.mode != "coach":
            raise ApiError("Not coaching.")
        self.stop.set()
        self.set_status("stopped", "Stopping after the current step...")

    def _load_demo(self, demo_path):
        """A robot demo CSV as is, or a recording turned into one with IK (cached in local/demos/)."""
        path = Path(demo_path)
        path = path if path.is_absolute() else REPO_ROOT / path
        if path.suffix == ".csv":
            return Demo.load_csv(str(path)).clean()

        cache = DEMO_CACHE / f"{path.stem}.csv"
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            self.set_status("preparing", f"Using the robot demo already made from {path.name}.")
            return Demo.load_csv(str(cache)).clean()

        from recording_demo import recording_to_demo

        def progress(done, total, failed):
            if self.stop.is_set():
                raise KeyboardInterrupt
            if done == 1 or done == total or done % 10 == 0:
                self.set_status("preparing", f"Turning {path.name} into a robot demo with IK: {done}/{total} poses"
                                + (f", {failed} out of reach" if failed else "") + ".")

        demo = recording_to_demo(path, self._ensure_ros_node(), on_progress=progress)
        failed, total = demo.meta["ik_failed"], demo.meta["ik_total"]
        if total == 0 or failed > MAX_UNREACHABLE * total:
            raise RuntimeError(f"{failed} of {total} poses in {path.name} are out of the robot's reach, so it can't "
                               "demonstrate it. If the robot moved backwards while recording, record again with "
                               "Flip ticked.")
        DEMO_CACHE.mkdir(parents=True, exist_ok=True)
        demo.save_csv(str(cache))
        return demo.clean()

    def _coach_thread(self, demo_path, robot_spec, reps, arm, speak):
        session = None
        try:
            robot = ReportingRobot(self._coach_robot(robot_spec), self)
            demo = self._load_demo(demo_path)
            kin = RobotKinematics()
            arm = arm or ("left" if demo.arm == "right" else "right")
            if self.stop.is_set():
                return
            if self.camera.fake:
                observer = FakeChild()
            else:
                self.camera_owner = "coach"
                observer = Observer(FrameReader(self.camera), arm=arm)
            session = SESSIONS / time.strftime("%Y%m%d-%H%M%S")
            self.set_status("waiting", f"Coaching '{demo.name}': robot {demo.arm} arm, your {arm} arm, {reps} reps.")
            coach = WebCoach(self, demo, robot, observer, kin, Voice(speak and not self.camera.fake),
                             record_s=5.0, session_dir=session, on_status=self._on_coach_status)
            coach.run(reps, pause_s=0.0 if self.camera.fake else 3.0)
        except KeyboardInterrupt:           # Stop pressed before Coach.run() had its own handler up
            pass
        except Exception as e:
            self.set_status("error", f"{type(e).__name__}: {e}")
            return
        finally:
            self.camera_owner = "preview"
            self.live = ""
            self.mode = "idle"
        if self.robot_error:
            self.set_status("error", f"The robot couldn't play the demo, so coaching stopped: {self.robot_error}")
        elif self.stop.is_set():
            self.set_status("stopped", "Coaching stopped.")
        else:
            scores = "  ".join(f"{r['score']:.0f}" for r in self.reps)
            self.set_status("finished", f"Scores: {scores}. Logged to local/sessions/{session.name}.")

    def _coach_robot(self, spec):
        if spec == "stub":
            return make_robot("stub", realtime=not self.camera.fake)
        if spec == "sim":
            # Only use a sim this server started: an unrelated server on a known port may be stale.
            if self.robot_state == "starting":
                self.set_status("starting", "Waiting for the simulated robot to finish starting in RViz...")
            return SocketRobot("127.0.0.1", self.ensure_robot(self.stop, announce=True))
        return make_robot(spec)

    def _start_sim(self, cancel, announce):
        """Start demo.launch.py (MoveIt, controllers, RViz) and play_demo.py's trajectory server.
        Call with _robot_lock held."""
        self._require_ros()
        self.robot_state = "starting"
        self.procs.stop(ROBOT_PROCESSES)    # a half-dead previous robot
        self.sim_port = None
        port = free_port()
        message = "Starting the simulated robot. RViz will open; this takes about 20 s."
        if announce:
            self.set_status("starting", message)
        else:
            self.log.append(f"{time.strftime('%H:%M:%S')}  {message}")
        self.procs.start("robot", ["ros2", "launch", "bracketbot_moveit_config", "demo.launch.py",
                                   f"rviz_config:={RVIZ_CONFIG}", "rviz_respawn:=true"])
        server, deadline = None, time.time() + 90.0
        while time.time() < deadline and not cancel.is_set():
            if server is None or server.poll() is not None:
                time.sleep(3.0)     # play_demo.py exits if the controllers aren't up yet; retry
                # run directly, not through `ros2 run`, which doesn't pass a signal on to the script
                server = self.procs.start("play_demo", ["python3", str(PLAY_DEMO), "--serve", str(port)])
            if SocketRobot("127.0.0.1", port).ping():
                self.sim_port = port
                self.robot_state = "ready"
                self.log.append(f"{time.strftime('%H:%M:%S')}  The simulated robot is ready in RViz.")
                return
            time.sleep(1.0)
        self.robot_state = "off" if cancel.is_set() else "error"
        if cancel.is_set():
            self.procs.stop(ROBOT_PROCESSES)
            raise KeyboardInterrupt
        raise RuntimeError("the simulated robot didn't come up within 90 s (see the log)")

    def _on_coach_status(self, phase, text, info):
        if phase == "adapting":
            reason = explain({**info, "message": text})
            self.reps.append({"rep": info["rep"], "score": info["score"], "sub": info["sub"],
                              "weakest": info["weakest"], "plan_before": info["plan_before"],
                              "plan_after": info["plan_after"], "message": text, "reason": reason})
            self.set_status(phase, f"{text} {reason}", info)
        else:
            self.set_status(phase, text, info)

    def shutdown(self):
        self.stop.set()
        self._closing.set()
        self.publishing = False
        if self._stopper is not None:
            self._stopper.join()
        self.procs.stop_all()
        if self._ros_node is not None:
            import rclpy
            self._ros_node.destroy_node()
            rclpy.try_shutdown()


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, payload):
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(200, app.snapshot())
            elif self.path.startswith("/stream.mjpg"):
                self._stream()
            else:
                self._json(404, {"error": "not found"})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    seq, jpeg = app.view.wait(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            routes = {
                "/api/camera": app.set_camera,
                "/api/record/start": app.record_start, "/api/record/stop": app.record_stop,
                "/api/coach/start": app.coach_start, "/api/coach/stop": app.coach_stop,
                "/api/mirror/start": app.mirror_start, "/api/mirror/stop": app.mirror_stop,
            }
            handler = routes.get(self.path)
            if handler is None:
                return self._json(404, {"error": "unknown endpoint"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, {"ok": True, "result": handler(**data)})
            except ApiError as e:
                self._json(409, {"error": str(e)})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="camera index, /dev/video path, stream URL, or 'fake'")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--demo-dir", action="append", default=[], metavar="DIR",
                    help="also offer the recordings (*.json) and robot demos (*.csv) in DIR; repeatable")
    ap.add_argument("--no-robot", action="store_true",
                    help="don't start the simulated robot and RViz with the server, only when a mode needs it")
    args = ap.parse_args()

    # Shut down through app.shutdown() (which stops the ROS processes) on Ctrl+C, a plain kill or the
    # terminal closing, including when started in the background, where SIGINT is ignored by default.
    # Anything a harder kill leaves running is stopped by the next server (Processes.reap_orphans).
    def interrupt(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)

    app = App(args.camera, args.demo_dir, start_robot=not args.no_robot,
              registry=LOCAL / f"webui_processes_{args.port}.json")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    print(f"web interface: http://localhost:{args.port}   (camera {args.camera})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        server.server_close()
        print("web interface stopped", flush=True)
        # Everything is cleaned up by now. A normal interpreter exit can still abort with
        # "terminate called without an active exception" while native OpenCV/MediaPipe threads run.
        os._exit(0)


if __name__ == "__main__":
    main()
