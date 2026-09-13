"""How the coach reaches the robot. Nothing here imports ROS.

    StubRobot    prints and waits — for developing on a laptop with no sim.
    SocketRobot  sends the trajectory as one JSON line to `play_demo.py --serve`
                 running on the sim machine, and waits for its reply.

Both expose `play(demo)` (blocking) and `play_async(demo)` (returns an Event
that is set when the robot has finished), so the coach can keep the webcam
running while the robot moves. Both take `on_event(name, seconds)`, called with
"approach" (gliding to the start pose), "start" (the demonstration itself) and
"return" (gliding back to the start pose).
"""

import json
import socket
import threading
import time
from typing import Callable, Optional

from .demo import Demo

OnEvent = Optional[Callable[[str, float], None]]


class Robot:
    def play(self, demo: Demo, on_event: OnEvent = None) -> float:
        """Play the demonstration; return the seconds it took."""
        raise NotImplementedError

    def play_async(self, demo: Demo, on_event: OnEvent = None) -> threading.Event:
        done = threading.Event()

        def run():
            try:
                self.play(demo, on_event)
            finally:
                done.set()

        threading.Thread(target=run, daemon=True).start()
        return done

    def close(self):
        pass


class StubRobot(Robot):
    """No robot: print the plan and (optionally) wait as long as it would take."""

    def __init__(self, realtime: bool = True, approach_s: float = 2.0, return_s: float = 2.0):
        self.realtime = realtime
        self.approach_s = approach_s
        self.return_s = return_s
        self.played = []

    def play(self, demo: Demo, on_event: OnEvent = None) -> float:
        m = demo.meta
        print(f"[stub robot] playing '{demo.name}': {len(demo)} pts, {demo.duration:.1f} s "
              f"(speed {m.get('speed', 1):.2f}, size {m.get('size', 1):.2f}, "
              f"segment {m.get('segment')}, hold {m.get('hold')})")
        self.played.append(demo)
        for name, seconds in (("approach", self.approach_s), ("start", demo.duration), ("return", self.return_s)):
            if on_event:
                on_event(name, seconds)
            if self.realtime:
                time.sleep(seconds)
        return self.approach_s + demo.duration + self.return_s


class SocketRobot(Robot):
    """Client for `play_demo.py --serve PORT` on the sim machine."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: Optional[float] = None):
        self.host, self.port, self.timeout = host, port, timeout

    def play(self, demo: Demo, on_event: OnEvent = None) -> float:
        t0 = time.time()
        timers, reply = [], {}
        with socket.create_connection((self.host, self.port), timeout=10.0) as s:
            s.settimeout(self.timeout or demo.duration + 30.0)
            s.sendall((demo.to_json() + "\n").encode())
            for line in s.makefile("rb"):
                msg = json.loads(line.decode() or "{}")
                if msg.get("event") != "accepted":
                    reply = msg
                    break
                # The server only replies at the end, so the phases are timed here from its durations.
                if on_event:
                    approach, duration, back = msg["approach"], msg["duration"], msg["return"]
                    on_event("approach", approach)
                    timers = [threading.Timer(approach, on_event, ("start", duration))]
                    if back > 0:
                        timers.append(threading.Timer(approach + duration, on_event, ("return", back)))
                    for timer in timers:
                        timer.daemon = True
                        timer.start()
        for timer in timers:
            timer.cancel()
        if not reply.get("ok"):
            raise RuntimeError(f"robot refused trajectory: {reply.get('error', reply)}")
        return time.time() - t0

    def ping(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=2.0) as s:
                s.sendall(b'{"cmd": "ping"}\n')
                return b"ok" in s.recv(1024)
        except OSError:
            return False


def make_robot(spec: str, realtime: bool = True) -> Robot:
    """'stub' | 'stub-fast' | 'host:port' -> Robot"""
    if spec == "stub":
        return StubRobot(realtime=realtime)
    if spec == "stub-fast":
        return StubRobot(realtime=False)
    host, _, port = spec.rpartition(":")
    return SocketRobot(host or "127.0.0.1", int(port))
