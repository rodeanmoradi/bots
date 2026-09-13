# webui

Web interface for recording, coaching and mirroring.

    webui/run.sh                                  # camera 0
    webui/run.sh --camera http://<phone-ip>:4747/video
    webui/run.sh --camera fake                    # no camera, simulated person
    webui/run.sh --demo-dir ~/demos               # extra demo folder
    webui/run.sh --no-robot                       # start the robot only when needed

Open http://localhost:8765 (works from Windows when running in WSL).

## Modes

- **Record**: tracks your arm; the robot mirrors you in RViz. Tick *Flip* if
  the robot moves backwards. Saves `local/recordings/<arm>_<view>_<time>.json`.
- **Coach**: the adaptive loop ([therapy/README.md](../therapy/README.md)).
  Logs to `local/sessions/<time>/`.
- **Mirror**: the robot copies you live.

## Behaviour

- **One camera connection**, shared by every mode.
- **Robot always running**: MoveIt, controllers and RViz start with the server.
  Closed RViz reopens (up to 10 times).
- **Modes start only their part**: Record/Mirror run `main_controller.py`;
  stopping sends the arm home.
- **Clean shutdown**: Ctrl+C, `kill` or closing the terminal stops everything.
  Leftovers from a crashed server are stopped by the next one.

## Demo list

Read from disk each time:

1. `local/recordings/*.json`, newest first (selected by default)
2. `therapy/demos/*.csv` and `*.csv` in the repo root
3. `--demo-dir` folders

Recordings are converted to joint-angle demos with IK when coaching starts and
cached in `local/demos/`. Coaching refuses a recording if over half its poses
are out of reach. Recordings need *simulated robot*.

## Files

| File | Role |
| --- | --- |
| `server.py` | HTTP server, camera, modes, ROS processes |
| `index.html` | the page |
| `recording_demo.py` | recording → joint demo via `/compute_ik` |
| `run.sh` | sources ROS, starts the server |

## API

- `GET /api/state`: everything the page shows
- `GET /stream.mjpg`: video
- `POST /api/camera`, `/api/record/start|stop`, `/api/coach/start|stop`, `/api/mirror/start|stop` (JSON body)
