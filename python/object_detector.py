import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
# All tuneable constants — change here rather than in class internals.

#VISUPATH_CAMERA_IDX       = 1  # not in use — find_camera_index() auto-detects cam index
VISUPATH_ALERT_INTERVAL_S = 1.5   # minimum seconds between consecutive spoken alerts
CONFIDENCE_THRESHOLD      = 0.45  # discard any detection below 45 % confidence
IOU_THRESHOLD             = 0.45  # NMS overlap threshold — higher keeps more overlapping boxes
GREY_PADDING_COLOR        = 114   # YOLOv8 canonical letterbox fill value (ImageNet mean ≈ 114)
HIGH_PRIORITY_OBJECTS     = {"traffic light", "stop sign"}  # always reported first if present

VLM_SNAPSHOT_INTERVAL_S = 30
VLM_SNAPSHOT_WIDTH = 640
VLM_SNAPSHOT_HEIGHT       = 480
VLM_SNAPSHOT_JPEG_QUALITY = 75


_HERE       = Path(__file__).parent
MODEL_PATH  = _HERE / "models" / "yolov8_quantized.tflite"
LABELS_PATH = _HERE / "coco_labels.txt"


def find_camera_index(start: int = 0, end: int = 10) -> int:
    """
    Scan camera indices [start, end) and return the first index that
    opens successfully AND returns a valid frame.
    Skips indices that open but can't read (e.g. metadata nodes like index 1).
    Raises RuntimeError if no usable camera is found.
    """
    print(f"[VisuPath] Scanning camera indices {start}–{end - 1}...")
    for idx in range(start, end):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        cap.release()

        if ok and frame is not None:
            print(f"[VisuPath]  Found usable camera at index {idx}")
            return idx

        print(f"[VisuPath] ⚠️  Index {idx} opened but returned no frame (metadata node?), skipping")

    raise RuntimeError(
        f"No usable camera found in index range {start}–{end - 1}. "
        "Check USB connection or extend the scan range."
    )


# Subset of COCO classes relevant to outdoor pedestrian navigation.
VISUPATH_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
    "dog", "chair", "couch", "bed", "dining table", "tv",
    "potted plant", "backpack", "suitcase", "microwave", "oven", "sink",
    "refrigerator",
}


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Detection:
    """Immutable result of a single object detection after all filtering."""

    label:      str
    confidence: float
    box:        tuple[float, float, float, float]  # (x1, y1, x2, y2) in frame pixels

    @property
    def area(self) -> float:
        """Pixel area of the bounding box — proxy for object distance."""
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center_x(self) -> float:
        """Horizontal centre of the box — used to determine left/centre/right direction."""
        x1, _, x2, _ = self.box
        return (x1 + x2) / 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_labels(path: str | os.PathLike[str]) -> list[str]:
    """Read one label per line; blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_interpreter(model_path: Path):
    """
    Load the TFLite interpreter, preferring the Qualcomm HTP delegate for faster
    inference on Snapdragon hardware; falls back to XNNPACK CPU if unavailable.
    Setting delegate=None in the except block prevents a spurious AttributeError
    from ai-edge-litert's __del__ on a half-constructed Delegate object.
    """
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError as exc:
            raise RuntimeError("Install ai-edge-litert before running.") from exc

    delegate = None
    try:
        delegate = tflite.load_delegate(
            # Absolute path required — relative paths silently fail on Linux.
            "/usr/local/lib/libQnnTFLiteDelegate.so",
            {
                "backend_type":         "htp",
                "htp_performance_mode": "2",  # sustained performance mode
                # htp_device_id and htp_precision removed — trigger stoi crash
                # on some QRB2210 runtime versions
            },
        )
        interpreter = tflite.Interpreter(
            model_path=str(model_path), experimental_delegates=[delegate]
        )
        print("✅ Using Qualcomm HTP delegate")

    except Exception as e:
        delegate = None  # prevents AttributeError in Delegate.__del__
        print(f"⚠️ HTP delegate unavailable ({e}), falling back to CPU")
        interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=4)

    interpreter.allocate_tensors()
    return interpreter


def _letterbox(
    frame: np.ndarray, tw: int, th: int, canvas: np.ndarray
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize `frame` to fit inside (tw × th) preserving aspect ratio, centred on a grey canvas.
    Reuses a pre-allocated `canvas` buffer to avoid per-frame heap allocation.
    Returns the filled canvas, the uniform scale factor, and (pad_x, pad_y) offsets
    needed to map model-space bounding boxes back to original frame coordinates.
    """
    h, w   = frame.shape[:2]
    scale  = min(tw / w, th / h)                        # uniform scale — no distortion
    rw, rh = int(round(w * scale)), int(round(h * scale))
    px, py = (tw - rw) // 2, (th - rh) // 2            # centre the image on the canvas

    canvas[:] = GREY_PADDING_COLOR                      # fill padding bands with grey
    canvas[py:py + rh, px:px + rw] = cv2.resize(
        frame, (rw, rh), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, (px, py)


# ── Detector ──────────────────────────────────────────────────────────────────

class YoloTfliteDetector:
    """
    Runs YOLOv8 inference on a uint8-quantized TFLite model with separate output
    tensors: boxes [1,8400,4], scores [1,8400], class_idx [1,8400].
    Outputs are dequantized via real = (raw - zero_point) * scale; class_idx is
    the exception (scale=0.0) — the raw uint8 value is the class index directly.
    """

    # Quantization params from interpreter.get_output_details() — hardcoded to avoid per-frame lookup.
    _BOX_SCALE,   _BOX_ZP   = 3.1009654998779297, 25   # boxes:     real = (raw - 25) * 3.1009
    _SCORE_SCALE, _SCORE_ZP = 0.00390625,          0   # scores:    real = raw * 0.00390625
    # class_idx: scale=0.0 → raw uint8 cast directly to int, no math needed

    def __init__(
        self,
        model_path:           Optional[Path] = None,
        labels_path:          Optional[Path] = None,
        confidence_threshold: float          = CONFIDENCE_THRESHOLD,
        iou_threshold:        float          = IOU_THRESHOLD,
    ):
        self.labels               = load_labels(labels_path or LABELS_PATH)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold

        self.interpreter    = _load_interpreter(model_path or MODEL_PATH)
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        _, self.input_height, self.input_width, _ = self.input_details[0]["shape"]

        # Pre-allocated buffers reused every frame to avoid per-frame heap pressure.
        self._canvas     = np.full(
            (self.input_height, self.input_width, 3),
            GREY_PADDING_COLOR, dtype=np.uint8,
        )
        self._rgb_buffer = np.empty_like(self._canvas)

        # Pre-compute valid class indices for a single vectorised np.isin() call per frame.
        self._valid_class_ids = np.array(
            [i for i, lbl in enumerate(self.labels) if lbl in VISUPATH_CLASSES],
            dtype=np.int32,
        )

        # Map output tensor name suffix → TFLite index; robust to reordered exports.
        self._out_idx = {
            info["name"].split("/")[-1]: info["index"]
            for info in self.output_details
        }
        print(f"[VisuPath] Output tensors: {list(self._out_idx)}")

        self._perf_frame_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Full inference pipeline for one frame.
        Returns Detection objects that passed confidence, class, and NMS filters.
        """

        # 1. Letterbox (BGR→RGB) — resize without distortion, pad with grey.
        t0 = time.perf_counter()

        canvas, scale, pad = _letterbox(
            frame, self.input_width, self.input_height, self._canvas
        )
        t_letterbox = time.perf_counter()

        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        t_pre = time.perf_counter()

        # 2. Inference — uint8 input maps directly to [0,1] float range, no scaling needed.
        self.interpreter.set_tensor(
            self.input_details[0]["index"], self._rgb_buffer[np.newaxis]
        )
        self.interpreter.invoke()
        t_infer = time.perf_counter()

        # 3. Dequantize outputs from uint8 to float32 (see class docstring).
        boxes_raw = (
            self.interpreter.get_tensor(self._out_idx["boxes"])[0].astype(np.float32)
            - self._BOX_ZP
        ) * self._BOX_SCALE                                     # → [8400, 4] float32

        scores = (
            self.interpreter.get_tensor(self._out_idx["scores"])[0].astype(np.float32)
            - self._SCORE_ZP
        ) * self._SCORE_SCALE                                   # → [8400] float32 in [0, 1]

        # class_idx: scale=0.0 — raw byte IS the COCO class index.
        class_ids = self.interpreter.get_tensor(
            self._out_idx["class_idx"]
        )[0].astype(np.int32)                                   # → [8400] int32
        t_dequant = time.perf_counter()

        # 4. Post-process and 5. log perf every 50 frames.
        result = self._postprocess(boxes_raw, scores, class_ids, frame.shape, scale, pad)
        t_post = time.perf_counter()

        self._perf_frame_count += 1
        if self._perf_frame_count % 50 == 0:
            ms = lambda a, b: f"{1000 * (b - a):.1f}ms"
            print(
                f"[Perf] frame={self._perf_frame_count} | "
                f"letterbox={ms(t0, t_letterbox)} | "
                f"bgr2rgb={ms(t_letterbox, t_pre)} | "
                f"infer={ms(t_pre, t_infer)} | "
                f"dequant={ms(t_infer, t_dequant)} | "
                f"post={ms(t_dequant, t_post)} | "
                f"total={ms(t0, t_post)}"
            )

        return result

    # ── Post-processing ───────────────────────────────────────────────────────

    def _postprocess(
        self,
        boxes_raw:   np.ndarray,
        scores:      np.ndarray,
        class_ids:   np.ndarray,
        frame_shape: tuple,
        scale:       float,
        pad:         tuple[int, int],
    ) -> list[Detection]:
        """
        Filter, deduplicate, and scale raw model outputs into Detection objects.
        Order: confidence filter → class filter → NMS → box scaling → object construction.
        Cheapest, highest-rejection filters run first to minimise NMS and loop overhead.
        """

        # Step 1: Confidence filter — eliminates ~99 % of 8400 anchors in one mask.
        conf_mask = scores >= self.confidence_threshold
        if not np.any(conf_mask):
            return []

        boxes_raw, scores, class_ids = (
            boxes_raw[conf_mask], scores[conf_mask], class_ids[conf_mask]
        )

        # Step 2: Class filter — keep only VISUPATH_CLASSES detections via vectorised isin.
        cls_mask = np.isin(class_ids, self._valid_class_ids)
        if not np.any(cls_mask):
            return []

        boxes_raw, scores, class_ids = (
            boxes_raw[cls_mask], scores[cls_mask], class_ids[cls_mask]
        )

        # Step 3: NMS — convert to [x,y,w,h] as required by cv2.dnn.NMSBoxes.
        x1 = boxes_raw[:, 0]
        y1 = boxes_raw[:, 1]
        boxes_xywh = np.stack(
            [x1, y1, boxes_raw[:, 2] - x1, boxes_raw[:, 3] - y1], axis=1
        )
        kept = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(),
            self.confidence_threshold, self.iou_threshold,
        )
        if len(kept) == 0:
            return []

        # reshape(-1) handles both OpenCV 4.x (nested) and 4.5+ (flat) return shapes.
        kept = np.asarray(kept).reshape(-1)
        boxes_raw, scores, class_ids = (
            boxes_raw[kept], scores[kept], class_ids[kept]
        )

        # Step 4: Reverse letterbox transform to get pixel coords in the original frame.
        fh, fw = frame_shape[:2]
        px, py = pad

        boxes_raw[:, [0, 2]] = np.clip((boxes_raw[:, [0, 2]] - px) / scale, 0, fw)
        boxes_raw[:, [1, 3]] = np.clip((boxes_raw[:, [1, 3]] - py) / scale, 0, fh)

        # Step 5: Build Detection objects for the small number of surviving boxes.
        return [
            Detection(
                label=self.labels[cid] if cid < len(self.labels) else f"class {cid}",
                confidence=float(conf),
                box=tuple(float(v) for v in box),
            )
            for box, conf, cid in zip(boxes_raw, scores, class_ids)
        ]


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ObjectDetectionPipeline:
    """
    Wraps YoloTfliteDetector in a camera capture loop and converts detections
    into human-readable alert strings for the TTS layer.
    Alerts are rate-limited by `alert_interval` to avoid repeated announcements.

    Optionally accepts `on_snapshot` callback invoked with a raw base64 JPEG
    string every VLM_SNAPSHOT_INTERVAL_S seconds. 
    """

    def __init__(self, on_snapshot: Optional[Callable[[str], None]] = None):
        self.detector       = YoloTfliteDetector()
        self.alert_interval = float(VISUPATH_ALERT_INTERVAL_S)
        self.camera_idx     = find_camera_index()
        self.last_alert     = 0.0
        self._on_snapshot = on_snapshot
        self._last_vlm_snap = 0.0

    def alerts(self) -> Iterable[str]:
        """
        Generator that yields alert strings when a relevant object is detected
        and the rate-limit window has elapsed. Yielded strings are the only values
        that reach the IPC queue; all print() calls go to stdout only.
        """
        camera = cv2.VideoCapture(self.camera_idx)

        # Buffer size 1 discards stale frames so we always process the most recent one.
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_idx}")

        cam_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[VisuPath] Camera opened: index={self.camera_idx}  {cam_w}×{cam_h}")
        print(
            f"[VisuPath] Running with "
            f"{self.detector.input_width}×{self.detector.input_height} model input"
        )

        frame_count  = 0
        t_start      = time.monotonic()
        LOG_INTERVAL = 100  # heartbeat every N frames

        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    # Transient read failure — retry rather than crash.
                    print("[VisuPath] ⚠️  camera.read() failed — retrying")
                    time.sleep(0.1)
                    continue

                frame_count += 1

                now = time.monotonic() # computed once, reused for both rate-limits

                # VLM snapshot: calls back into main.py
                if self._on_snapshot is not None and now - self._last_vlm_snap >= VLM_SNAPSHOT_INTERVAL_S:
                    self._last_vlm_snap = now
                    self._fire_snapshot(frame)


                detections = self.detector.detect(frame)

                # Heartbeat: log throughput and detection count every LOG_INTERVAL frames.
                if frame_count % LOG_INTERVAL == 0:
                    elapsed = time.monotonic() - t_start
                    fps     = frame_count / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[VisuPath] Frame {frame_count} | "
                        f"{fps:.1f} fps | "
                        f"detections this frame: {len(detections)}"
                    )

                if detections:
                    summary = ", ".join(
                        f"{d.label} ({round(d.confidence * 100)}%)"
                        for d in detections
                    )
                    print(f"[VisuPath] Detected: {summary}")

                # Rate-limited alert — only the yield feeds the IPC queue.
                alert = self._build_alert(detections, frame.shape)
                
                if alert and now - self.last_alert >= self.alert_interval:
                    self.last_alert = now
                    print(f"[VisuPath] Alert → {alert}")
                    yield alert

        finally:
            camera.release()
            elapsed = time.monotonic() - t_start
            print(
                f"[VisuPath] Camera released after {frame_count} frames "
                f"({frame_count / max(elapsed, 1e-6):.1f} fps avg)"
            )

    def _fire_snapshot(self, frame: np.ndarray) -> None:
        """
        Encode the current frame as a base64 JPEG and invoke on_snapshot.
        """
        t0 = time.perf_counter()

        resized = cv2.resize(
            frame,
            (VLM_SNAPSHOT_WIDTH, VLM_SNAPSHOT_HEIGHT),
            interpolation = cv2.INTER_LINEAR,
        )

        ok, buf = cv2.imencode(
            ".jpg", resized,
            [cv2.IMWRITE_JPEG_QUALITY, VLM_SNAPSHOT_JPEG_QUALITY],
        )

        if not ok:
            print("[Visupath] VLM snapshot encode failed: skipping")
            return

        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        self._on_snapshot(image_b64) # callback (main.py will put it in the scene_queue to send to phone)

        elapsed_ms = 1000 * (time.perf_counter() - t0)

        print(f"[Visupath] VLM snapshot fired encode={elapsed_ms:.1f}ms payload={len(image_b64)}chars")



    def _build_alert(
        self, detections: list[Detection], frame_shape: tuple
    ) -> Optional[str]:
        """
        Format the most important detection as a spoken alert string.
        Safety-critical objects (traffic light, stop sign) take priority;
        otherwise the largest box (≈ closest object) is chosen.
        """
        if not detections:
            return None

        safety = [d for d in detections if d.label in HIGH_PRIORITY_OBJECTS]
        best   = max(safety or detections, key=lambda d: d.area)

        fh, fw = frame_shape[:2]
        third  = fw / 3

        # Divide frame into left/centre/right thirds for direction.
        direction = (
            "to your left"  if best.center_x < third else
            "to your right" if best.center_x > third * 2 else
            "ahead"
        )

        # Box height relative to frame height as a rough distance proxy.
        _, y1, _, y2 = best.box
        ratio    = (y2 - y1) / max(fh, 1)
        distance = (
            "very close"   if ratio > 0.55 else
            "nearby"       if ratio > 0.30 else
            "farther away"
        )

        return (
            f"{best.label} detected {direction}, "
            f"{distance})"
        )