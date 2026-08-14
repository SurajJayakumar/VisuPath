# [VisuPath](https://projecthub.arduino.cc/projects/4d0c2cad-f592-40fe-979d-664cce0d2ea0/preview)

<img width="740" height="1125" alt="image" src="https://github.com/user-attachments/assets/4f470c5e-575f-4dfe-8b02-c28f64018a31" />
<img width="800" height="634" alt="image" src="https://github.com/user-attachments/assets/3d7acfe8-808d-4926-a71b-5805dd5bf8d0" />

Smart mobility cane for visually impaired users. Has blazing fast real-time
distance sensing and haptic feedback. The Qualcomm MPU runs YOLOv8 object
detection and sends alerts to your phone over Wifi (WebSocket) with low latency (~0.2s). There is also a worker process on the phone that sends images intermittently to an AI model (VLM) for inferencing to explain the current scene.

Not to mention Text-to-Speech, so you can hear the alerts & react immediately!

In this way:

1) Your cane vibrates when there are obstacles close by.

2) Your phone tells you what obstacles are around you + how far away they are + direction / trajectory.

3) Your phone gives you an in depth explanation of 
the current scene every minute.

---

## Features

- **Voice Alerts**: spoken object alerts: "chair detected ahead, nearby"
- **Haptic Feedback**: vibration intensity scales linearly with proximity (off > 1100 mm, max < 300 mm)
- **Gemini VLM Scene Understanding**: every 60 s a camera frame is sent to Gemini, which returns a structured natural-language description: *"Environment: indoor corridor. Hazard: chair ahead to the right."* Spoken at higher priority than YOLO alerts. Observed end-to-end latency ~18 s, well within the 60 s snapshot window.
- **Two-Tier Detection**: YOLOv8 for immediate per-object alerts; Gemini VLM for periodic scene context. YOLO tells you *what* is there; Gemini tells you *where you are*.
- **Object Awareness**: 80 COCO classes: people, vehicles, furniture, traffic signs, and more

---

## Architecture

```text
Arduino Uno Q
├── STM32 MCU
│   └── sketch/sketch.ino: distance sensing, haptic feedback
│
└── Qualcomm MPU / Debian
    └── python/main.py
        └── WebSocket over Wi-Fi
            └── Phone / Termux
                ├── phone/server.js      : WebSocket relay
                ├── phone/vlm_worker.py  : Gemini cloud VLM
                └── phone/tts_client.py  : text-to-speech
```

The board connects **to** the phone as a WebSocket client. The phone runs the server.

---

## Project Files

### `sketch/`: STM32 MCU

| File | Purpose |
|---|---|
| `sketch/sketch.ino` | Polls VL53L4CD distance sensor. Drives Modulino Vibro with intensity proportional to proximity. |
| `sketch/sketch.yaml` | Arduino project profile (`arduino:zephyr` platform, Modulino/VL53L4 libraries). |

### `python/`: Qualcomm MPU (Linux)

| File | Purpose |
|---|---|
| `python/main.py` | Entry point. Spawns WebSocket client process, runs YOLO detection loop, manages IPC queues and shutdown. |
| `python/object_detector.py` | YOLOv8 TFLite pipeline. Qualcomm HTP delegate (CPU fallback), NMS, alert generation, VLM snapshot every 60 s. |
| `python/board_websocket_client.py` | Async WebSocket client to phone `/board`. Sends `0x01` YOLO text and `0x02` base64 JPEG. Auto-reconnects. |
| `python/coco_labels.txt` | COCO class names for the YOLO model. |

### `phone/`: Phone / Termux

| File | Purpose |
|---|---|
| `phone/server.js` | Node.js WebSocket server (port 8765). Routes `0x01` alerts to `tts.sock`, `0x02` snapshots to `vlm.sock`. Drops VLM requests while inference is in flight. |
| `phone/vlm_worker.py` | Gemini VLM worker. Listens on `vlm.sock`. Posts base64 JPEG to Gemini HTTPS API. Fallback model chain on 429/503/timeout. Errors never forwarded to TTS. |
| `phone/tts_client.py` | TTS client. Listens on `tts.sock`. Speaks via `termux-tts-speak`. VLM scene descriptions (`0x02`) spoken before YOLO alerts (`0x01`). Latest message per tier only: no backlog. |
| `phone/socket_server.py` | Shared Unix socket boilerplate (`serve_unix_socket`, `read_lines`). |
| `phone/start.sh` | Launches `vlm_worker.py`, `tts_client.py`, and `server.js`. Ctrl+C kills all three cleanly. |

### `visupath.service`: Uno Q systemd unit

Auto-starts `main.py` after network comes up. Restarts on failure (5 s delay). Logs to journal.

---

## Setup

### 1. Wi-Fi

Both devices must be on the same network. Recommended: phone hotspot.

Enable hotspot on the phone: `Settings → Network & internet → Hotspot & tethering → Wi-Fi hotspot`

Connect the Uno Q to the same wifi as phone (phone hotspot if needed) using arduino App Lab.

The board connects **to** the phone. Set `PHONE_IP` in `python/board_websocket_client.py` to the phone's hotspot gateway. 
1 way to get the phone's IP address is by installing termux (steps below), and running `ifconfig`.

---

### 2. Phone / Termux Setup

Install both from **F-Droid** (not the Play Store: Play Store builds are outdated):

- Termux: https://f-droid.org/packages/com.termux/
- Termux:API: https://f-droid.org/packages/com.termux.api/

> **Termux:API is required.** `termux-tts-speak` is provided by Termux:API. Without it, TTS fails silently. Grant it microphone and notification permissions in Android Settings.

```bash
pkg update && pkg upgrade
pkg install python nodejs git

cd ~
git clone https://github.com/SurajJayakumar/VisuPath.git visupath
```

Alternatively, if you already have `sshd` running on the phone (see SSH section below), copy the `phone/` directory from your development machine:

```bash
# Run from your dev machine
scp -P 8022 -r phone/ <PHONE_IP>:~/visupath/
```

Then inside Termux, set up the venv:

```bash
cd ~/visupath
python -m venv .venv
source .venv/bin/activate
pip install requests
npm install ws

chmod +x ~/visupath/start.sh
```

Optional auto-start on reboot: install Termux:Boot (F-Droid), create `~/.termux/boot/start-visupath` that runs `./start.sh`, and disable battery optimization for Termux in Android Settings.

**SSH access (optional)**: useful for managing the phone remotely from the Uno Q or a laptop on the same network:

```bash
pkg install openssh
passwd        # set a password for the Termux user
sshd          # starts SSH server on port 8022
```

Connect from another machine:

```bash
ssh -p 8022 <PHONE_IP>
```

Run `sshd` again after each Termux restart, or add it to the Termux:Boot script.

---

### 3. Gemini API

```bash
echo 'export GEMINI_API_KEY="YOUR_API_KEY_HERE"' >> ~/.bashrc
echo 'export GEMINI_MODEL="gemini-3.5-flash"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
python - <<'PY'
import os
print("key loaded:", bool(os.environ.get("GEMINI_API_KEY")))
print("model:", os.environ.get("GEMINI_MODEL"))
PY
```

Working model: `gemini-3.5-flash`. Do not use `gemini-2.0-flash-lite-001`, `gemini-2.5-flash`, or `gemini-3.5-flash-lite`: these produced errors for this account.

---

### 4. Arduino Uno Q

**Flash the sketch:** open `sketch/sketch.ino` in Arduino App Lab and flash to the STM32 MCU. Confirm distance readings appear in the Bridge monitor.

**Set up the Linux-side project** at `/home/arduino/visupath_python/`:

```bash
# Verify path
find /home/arduino -name main.py 2>/dev/null

cd /home/arduino/visupath_python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Install the YOLO model**: export a YOLOv8 Detection TFLite model from https://aihub.qualcomm.com/models/yolov8_det and place it at:

```
/home/arduino/visupath_python/models/yolov8_quantized.tflite
```

**Test manually:**

```bash
cd /home/arduino/visupath_python
source .venv/bin/activate
python3 main.py
```

---

### 5. Uno Q systemd Service

#### Install and enable

```bash
sudo cp visupath.service /etc/systemd/system/visupath.service
sudo systemctl daemon-reload
sudo systemctl enable visupath.service   # register for auto-start on every boot
sudo systemctl start visupath.service    # start immediately (without rebooting)
```

#### Lifecycle commands

| Command | Effect |
|---|---|
| `sudo systemctl start visupath.service` | Start the service now |
| `sudo systemctl stop visupath.service` | Stop the running process |
| `sudo systemctl restart visupath.service` | Stop then start. use after editing `main.py` or `visupath.service` |
| `sudo systemctl enable visupath.service` | Register for auto-start on boot |
| `sudo systemctl disable visupath.service` | Remove from boot (does not stop a running instance) |
| `systemctl status visupath.service` | Show current state, PID, and last log lines |
| `journalctl -u visupath.service -f` | Stream live logs. primary debugging tool |

#### Arduino App Lab CLI

If you manage the app through Arduino App Lab rather than systemd directly, use `arduino-app-cli`:

```bash
arduino-app-cli app list                   # list deployed apps and their status
arduino-app-cli app start visupath         # start the app
arduino-app-cli app stop visupath          # stop the app
arduino-app-cli app restart visupath       # restart after changes
arduino-app-cli app logs visupath          # stream logs
arduino-app-cli app enable visupath        # enable auto-start on boot
arduino-app-cli app disable visupath       # disable auto-start
```

---

### 6. Start the Phone

```bash
cd ~/visupath
source .venv/bin/activate
./start.sh
```

Expected output:

```
[VisuPath] Using Gemini model: gemini-3.5-flash
[VisuPath] Starting VLM worker...
[VisuPath] Starting TTS client...
[VisuPath] Starting WebSocket server...

=== VisuPath running ===
  vlm_worker : PID ...
  tts_client : PID ...
  server.js  : PID ...

Press Ctrl+C to stop all.
```

---

## Running

Startup order:

```
1. Enable phone hotspot
2. Uno Q connects to Wi-Fi (NetworkManager, automatic)
3. Uno Q systemd service starts main.py
4. On phone: cd ~/visupath && ./start.sh
5. Board WebSocket client connects to phone server.js
6. YOLO alerts stream to TTS
7. VLM snapshots sent to Gemini every 60 s; scene descriptions spoken first
```

---

## Testing Checklist

**Uno Q**

```bash
systemctl status visupath.service
journalctl -u visupath.service -f
ping -c 3 <PHONE_IP>
```

- [ ] Wi-Fi connected
- [ ] MCU sketch running (distance readings in Bridge monitor)
- [ ] Camera opened, YOLO detections in journal
- [ ] `[Websocket Client] Connected to phone!`

**Phone**

```bash
cd ~/visupath && source .venv/bin/activate
echo "$GEMINI_MODEL"
python -c "import requests; print('requests OK')"
./start.sh
```

- [ ] `[VLM Worker] Listening on ./vlm.sock`
- [ ] `[TTS Client] Listening on ./tts.sock`
- [ ] `[VisuPath] WS server running on port 8765`

**End-to-end**

```
[Board] Arduino connected
[VLM] Snapshot dispatched
[Board] Alert: chair detected ahead, nearby
[TTS] Spoke in ...ms: chair detected ahead, nearby
```

---

## FAQ

**How is distance estimated?**
Bounding-box height relative to frame height (`nearby` / `very close` / `farther away`). Not a depth sensor.

**What drives the vibration?**
The ToF distance sensor on the MCU. Intensity scales linearly with closeness, independent of YOLO.

**What if Wi-Fi drops?**
The WebSocket client reconnects automatically.

**Can I tune alert frequency?**
Yes: edit the constants at the top of `python/object_detector.py` (`VISUPATH_ALERT_INTERVAL_S`, `VLM_SNAPSHOT_INTERVAL_S`, etc.).

**What if a YOLO alert and a scene description arrive at the same time?**
VLM scene descriptions are always spoken first. Only the latest message per tier is kept: no backlog.

**Does Gemini store my images?**
Each request is independent. The worker does not maintain a session; no history is retained.

---

## Future Work

| Priority | Item |
|---|---|
| 1 | On-phone multimodal VLM via MediaPipe / LiteRT (requires Android app); Gemini as cloud fallback |
| 2 | Monocular depth + ToF sensor fusion for metric distance (Testing in progress)|
| 3 | Approach velocity estimation and safety-aware alert prioritisation |
| 4 | Camera pan/tilt motion compensation in object tracking |
| 5 | Calibration pipeline (FOV, lens distortion, class-specific size priors) |
| 6 | MCU sketch parsing YOLO alerts for richer haptic feedback patterns |
| 7 | IoU-based or Kalman-filter object tracking (replacing size-based matching) |
| 8 | Layered inference: local safety alerts → on-device VLM → Gemini fallback |
