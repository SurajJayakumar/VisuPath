# 🦯 VisuPath

A smart mobility cane that tells you what's around you, before you walk into it.

VisuPath is being built on Arduino UNO Q. The Linux side runs camera object
detection and sends spoken-alert text to a phone/browser over WebSocket. The
sketch side can drive haptic feedback through a Modulino Vibro.

---

## What It Does

VisuPath gives you a real-time picture of your surroundings through vibrations and
voice alerts on your phone. No screen needed. No hands needed. Just walk.

---

## Features

- 🔊 **Voice Alerts** — Hear what's ahead: "Car approaching, 5 feet away"
- 📳 **Haptic Feedback** — Cane vibrates harder and faster as obstacles get closer
- 🚗 **Speed Awareness** — Detects if objects are moving fast (car, bike) or slow (person, dog)
- 🌍 **Panoramic Scene Understanding** — Builds awareness of everything around you, not just straight ahead
- 🚦 **Intersection Intelligence** — Detects if the walk sign says GO or STOP before you step off the curb
- 🚗 **Turning Car Detection** — Warns you if a car is turning into your path at a red light
- 📸 **Fast Sign Reading** — Quickly identifies traffic signals at low resolution for speed
- 🐕 **Object Awareness** — Recognizes trees, people, dogs, cars, chairs, tables, fire hydrants and more

---

## Why It Matters

| Situation | Problem | VisuPath | Outcome |
|---|---|---|---|
| 🚶 Sidewalk | 🌿 Can't see a bush near your shoulder until you hit it | Cane vibrates, voice says "foliage ahead to your right, 2 feet" | You step around it safely |
| 🚦 Intersection | Don't know if it's safe to cross | Detects walk sign, says "Walk sign is ON" | You cross with confidence |
| 🚗 Car turning right on red | Silent fast-moving threat | Detects moving bounding box, alerts immediately ⚡| You stop before stepping off curb 🛑|
| 🚲Person biking towards you | Unpredictable fast movement | Speed + proximity triggers urgent haptic pulse 📳 | You redirect 💪|
| 🏫 Unfamiliar indoor space | Chairs and tables everywhere 🪑🗄️| Full scene scan identifies all objects and positions | You navigate the room safely 🗺️ |

---

## Current Implementation

The current prototype implements object-alert text generation and phone/browser
delivery. Some product features above are still roadmap items.

```text
USB/CSI camera
  -> Python YOLOv8 object detection
  -> alert text queue
  -> WebSocket /board
  -> Node relay
  -> WebSocket /tts clients
```

## Object Detection

The Python app expects a YOLOv8 object-detection TFLite model exported from
Qualcomm AI Hub. Qualcomm's YOLOv8-Detection model uses a 640x640 input and is
intended for mobile and edge deployment.

Model page:

```text
https://aihub.qualcomm.com/models/yolov8_det
```

Use AI Hub to export a deployable TFLite model, then copy
that exported file into this project.

Place the exported model in one of these locations:

```text
models/yolov8_det_quantized.tflite
models/YOLOv8-Detection.tflite
YOLOv8-Detection.tflite
```

Or point to it explicitly:

```bash
export VISUPATH_MODEL_PATH=/path/to/yolov8_det_quantized.tflite
```

Optional runtime settings:

```bash
export VISUPATH_CAMERA_INDEX=0
export VISUPATH_ALERT_INTERVAL=1.5
export VISUPATH_LABELS_PATH=python/coco_labels.txt
```

`python/coco_labels.txt` contains the COCO (common objects in context) class names used by the YOLO model.
The detector uses it to turn numeric class IDs into readable alert text such as
`person`, `bicycle`, or `car`.

## Install

On the UNO Q Linux side:

```bash
pip install -r requirements.txt
```

If `tflite-runtime` is not available from pip for your board image, install the
TensorFlow Lite runtime package recommended by the UNO Q/App Lab image and keep
the Python import path available to this app.

## Run

Start the phone/browser relay:

```bash
node server.js
```

Then run detection:

```bash
python python/main.py
```

The phone/browser client should connect to:

```text
ws://<UNO_Q_IP>:8765/tts
```

## Notes

- The current distance wording is based on bounding-box size, not a depth sensor.
- The Arduino sketch reads the Modulino Distance sensor and prints `TOO_CLOSE`
  messages when an obstacle is within 700 mm. It also pulses the Modulino Vibro
  harder as the obstacle gets closer.
- The Arduino sketch is not yet parsing YOLO alert messages from the Linux side.
- The WebSocket sender automatically reconnects to the relay when the connection
  drops.
