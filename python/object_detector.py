import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


DEFAULT_MODEL_PATHS = (
    Path("models/yolov8_det_quantized.tflite"),
    Path("models/YOLOv8-Detection.tflite"),
    Path("YOLOv8-Detection.tflite"),
)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center_x(self) -> float:
        x1, _, x2, _ = self.box
        return (x1 + x2) / 2

    @property
    def center_y(self) -> float:
        _, y1, _, y2 = self.box
        return (y1 + y2) / 2

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.box
        return max(0.0, x2 - x1)

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.box
        return max(0.0, y2 - y1)


@dataclass(frozen=True)
class DepthEstimate:
    detection: Detection
    distance_m: Optional[float]
    radial_velocity_mps: Optional[float]

    @property
    def area(self) -> float:
        return self.detection.area

    @property
    def label(self) -> str:
        return self.detection.label

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def center_x(self) -> float:
        return self.detection.center_x

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.detection.box


# Approximate physical object dimensions for COCO labels, in meters.
# These are priors for monocular depth, not guarantees. Calibrate focal length
# for the target camera and environment before using alerts for safety decisions.
OBJECT_DIMENSIONS_M = {
    "person": (0.45, 1.70),
    "bicycle": (1.70, 1.05),
    "car": (1.80, 1.50),
    "motorcycle": (2.10, 1.20),
    "airplane": (35.0, 12.0),
    "bus": (2.55, 3.10),
    "train": (3.00, 4.10),
    "truck": (2.60, 3.20),
    "boat": (2.00, 1.50),
    "traffic light": (0.30, 0.90),
    "fire hydrant": (0.30, 0.70),
    "stop sign": (0.75, 0.75),
    "parking meter": (0.30, 1.40),
    "bench": (1.50, 0.90),
    "bird": (0.25, 0.20),
    "cat": (0.45, 0.30),
    "dog": (0.75, 0.60),
    "horse": (2.20, 1.60),
    "sheep": (1.20, 0.90),
    "cow": (2.40, 1.50),
    "elephant": (5.50, 3.20),
    "bear": (1.40, 1.20),
    "zebra": (2.40, 1.40),
    "giraffe": (2.30, 4.80),
    "backpack": (0.35, 0.50),
    "umbrella": (1.00, 1.00),
    "handbag": (0.35, 0.30),
    "suitcase": (0.45, 0.65),
    "sports ball": (0.22, 0.22),
    "skateboard": (0.80, 0.20),
    "chair": (0.50, 0.90),
    "couch": (2.00, 0.85),
    "potted plant": (0.45, 0.80),
    "bed": (2.00, 0.70),
    "dining table": (1.50, 0.75),
    "toilet": (0.40, 0.75),
    "tv": (1.00, 0.60),
    "laptop": (0.35, 0.25),
    "microwave": (0.50, 0.30),
    "oven": (0.60, 0.60),
    "sink": (0.55, 0.25),
    "refrigerator": (0.85, 1.75),
    "book": (0.20, 0.28),
    "vase": (0.25, 0.35),
}


def resolve_model_path() -> Path:
    configured = os.getenv("VISUPATH_MODEL_PATH")
    candidates = [Path(configured)] if configured else list(DEFAULT_MODEL_PATHS)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No YOLO model file found. Export/download a Qualcomm AI Hub YOLOv8 "
        f"TFLite model and set VISUPATH_MODEL_PATH, or place it at: {searched}"
    )


def load_labels(path: str | os.PathLike[str]) -> list[str]:
    with open(path, "r", encoding="utf-8") as label_file:
        return [line.strip() for line in label_file if line.strip()]


def _load_interpreter(model_path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Install tflite-runtime on the UNO Q, or install TensorFlow "
                "for local development, before running object detection."
            ) from exc

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, (pad_x, pad_y)


def _deletterbox(
    box: Iterable[float],
    frame_shape: tuple[int, int, int],
    scale: float,
    pad: tuple[int, int],
) -> tuple[float, float, float, float]:
    pad_x, pad_y = pad
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = min(max((x1 - pad_x) / scale, 0), frame_width)
    y1 = min(max((y1 - pad_y) / scale, 0), frame_height)
    x2 = min(max((x2 - pad_x) / scale, 0), frame_width)
    y2 = min(max((y2 - pad_y) / scale, 0), frame_height)
    return x1, y1, x2, y2


class YoloTfliteDetector:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
    ):
        self.model_path = model_path or resolve_model_path()
        self.labels = load_labels(labels_path or Path(__file__).with_name("coco_labels.txt"))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.interpreter = _load_interpreter(self.model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.input_height, self.input_width, _ = self.input_details[0]["shape"]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        input_frame, scale, pad = _letterbox(frame, int(self.input_width))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_info = self.input_details[0]
        input_data = self._prepare_input(input_frame, input_info)

        self.interpreter.set_tensor(input_info["index"], input_data)
        self.interpreter.invoke()

        outputs = [
            self._read_output(output_info)
            for output_info in self.output_details
        ]
        return self._postprocess(outputs, frame.shape, scale, pad)

    def _prepare_input(self, input_frame: np.ndarray, input_info: dict) -> np.ndarray:
        input_data = input_frame[np.newaxis, ...]

        if input_info["dtype"] == np.float32:
            return input_data.astype(np.float32) / 255.0

        scale, zero_point = input_info.get("quantization", (0.0, 0))
        if scale:
            real_input = input_data.astype(np.float32)
            if scale < 0.01:
                real_input = real_input / 255.0
            input_data = np.round(real_input / scale + zero_point)

        dtype = input_info["dtype"]
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            input_data = np.clip(input_data, limits.min, limits.max)

        return input_data.astype(dtype)

    def _read_output(self, output_info: dict) -> np.ndarray:
        output = self.interpreter.get_tensor(output_info["index"])
        scale, zero_point = output_info.get("quantization", (0.0, 0))
        if scale and output.dtype != np.float32:
            return (output.astype(np.float32) - zero_point) * scale
        return output

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        frame_shape: tuple[int, int, int],
        scale: float,
        pad: tuple[int, int],
    ) -> list[Detection]:
        predictions = np.squeeze(outputs[0])
        if predictions.ndim != 2:
            return []

        if predictions.shape[0] in (84, 85):
            predictions = predictions.T

        boxes_for_nms = []
        detections = []
        input_size = float(self.input_width)

        for prediction in predictions:
            if prediction.shape[0] == 6:
                x1, y1, x2, y2, confidence, class_id = prediction
            else:
                objectness = float(prediction[4]) if prediction.shape[0] == 85 else 1.0
                class_scores = prediction[5:] if prediction.shape[0] == 85 else prediction[4:]
                class_id = int(np.argmax(class_scores))
                confidence = objectness * float(class_scores[class_id])
                if confidence < self.confidence_threshold:
                    continue

                cx, cy, width, height = prediction[:4]
                x1 = cx - width / 2
                y1 = cy - height / 2
                x2 = cx + width / 2
                y2 = cy + height / 2

            if confidence < self.confidence_threshold:
                continue

            if max(x1, y1, x2, y2) <= 1.5:
                x1, y1, x2, y2 = x1 * input_size, y1 * input_size, x2 * input_size, y2 * input_size

            x1, y1, x2, y2 = _deletterbox((x1, y1, x2, y2), frame_shape, scale, pad)
            label = self.labels[int(class_id)] if int(class_id) < len(self.labels) else f"class {int(class_id)}"
            detections.append(Detection(label=label, confidence=float(confidence), box=(x1, y1, x2, y2)))
            boxes_for_nms.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        if not detections:
            return []

        indexes = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            [detection.confidence for detection in detections],
            self.confidence_threshold,
            self.iou_threshold,
        )
        kept = indexes.flatten().tolist() if len(indexes) else []
        return [detections[index] for index in kept]


class MonocularDepthTracker:
    def __init__(
        self,
        horizontal_fov_deg: float = 62.0,
        focal_length_px: Optional[float] = None,
        max_match_distance_ratio: float = 0.18,
        smoothing_alpha: float = 0.45,
        stale_after_s: float = 2.0,
    ):
        self.horizontal_fov_deg = horizontal_fov_deg
        self.configured_focal_length_px = focal_length_px
        self.max_match_distance_ratio = max_match_distance_ratio
        self.smoothing_alpha = smoothing_alpha
        self.stale_after_s = stale_after_s
        self._tracks: dict[int, dict[str, float | str]] = {}
        self._next_track_id = 1

    def estimate(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int, int],
        timestamp: Optional[float] = None,
    ) -> list[DepthEstimate]:
        timestamp = timestamp if timestamp is not None else time.monotonic()
        self._drop_stale_tracks(timestamp)

        frame_height, frame_width = frame_shape[:2]
        focal_length_px = self._focal_length_px(frame_width)
        estimates = []
        matched_track_ids: set[int] = set()

        for detection in sorted(detections, key=lambda item: item.area, reverse=True):
            distance_m = self._distance_m(detection, focal_length_px)
            track_id = self._match_track(detection, frame_width, frame_height, matched_track_ids)
            radial_velocity_mps = None

            if track_id is not None:
                track = self._tracks[track_id]
                previous_distance = track.get("distance_m")
                previous_timestamp = float(track["timestamp"])
                dt = max(timestamp - previous_timestamp, 0.001)

                if distance_m is not None and previous_distance is not None:
                    raw_velocity = (distance_m - float(previous_distance)) / dt
                    previous_velocity = track.get("radial_velocity_mps")
                    if previous_velocity is None:
                        radial_velocity_mps = raw_velocity
                    else:
                        radial_velocity_mps = (
                            self.smoothing_alpha * raw_velocity
                            + (1.0 - self.smoothing_alpha) * float(previous_velocity)
                        )

                self._update_track(track_id, detection, timestamp, distance_m, radial_velocity_mps)
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._update_track(track_id, detection, timestamp, distance_m, None)

            matched_track_ids.add(track_id)
            estimates.append(DepthEstimate(detection, distance_m, radial_velocity_mps))

        return estimates

    def _focal_length_px(self, frame_width: int) -> float:
        if self.configured_focal_length_px:
            return self.configured_focal_length_px

        fov_rad = np.deg2rad(max(1.0, min(self.horizontal_fov_deg, 179.0)))
        return frame_width / (2.0 * np.tan(fov_rad / 2.0))

    def _distance_m(self, detection: Detection, focal_length_px: float) -> Optional[float]:
        dimensions = OBJECT_DIMENSIONS_M.get(detection.label)
        if dimensions is None:
            dimensions = (
                float(os.getenv("VISUPATH_DEFAULT_OBJECT_WIDTH_M", "0.5")),
                float(os.getenv("VISUPATH_DEFAULT_OBJECT_HEIGHT_M", "1.0")),
            )

        real_width_m, real_height_m = dimensions
        estimates = []
        if detection.height >= 6:
            estimates.append(real_height_m * focal_length_px / detection.height)
        if detection.width >= 6:
            estimates.append(real_width_m * focal_length_px / detection.width)

        if not estimates:
            return None

        return float(np.median(estimates))

    def _match_track(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int,
        matched_track_ids: set[int],
    ) -> Optional[int]:
        best_track_id = None
        best_score = float("inf")
        max_distance_px = self.max_match_distance_ratio * max(frame_width, frame_height)

        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids or track["label"] != detection.label:
                continue

            dx = float(track["center_x"]) - detection.center_x
            dy = float(track["center_y"]) - detection.center_y
            center_distance = float(np.hypot(dx, dy))
            if center_distance > max_distance_px:
                continue

            area = max(float(track["area"]), detection.area, 1.0)
            area_change = abs(float(track["area"]) - detection.area) / area
            score = center_distance + area_change * max_distance_px
            if score < best_score:
                best_track_id = track_id
                best_score = score

        return best_track_id

    def _update_track(
        self,
        track_id: int,
        detection: Detection,
        timestamp: float,
        distance_m: Optional[float],
        radial_velocity_mps: Optional[float],
    ) -> None:
        self._tracks[track_id] = {
            "label": detection.label,
            "center_x": detection.center_x,
            "center_y": detection.center_y,
            "area": detection.area,
            "timestamp": timestamp,
            "distance_m": distance_m,
            "radial_velocity_mps": radial_velocity_mps,
        }

    def _drop_stale_tracks(self, timestamp: float) -> None:
        stale_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp - float(track["timestamp"]) > self.stale_after_s
        ]
        for track_id in stale_track_ids:
            del self._tracks[track_id]


class ObjectDetectionPipeline:
    def __init__(self):
        labels_path = Path(os.getenv("VISUPATH_LABELS_PATH", Path(__file__).with_name("coco_labels.txt")))
        self.detector = YoloTfliteDetector(labels_path=labels_path)
        focal_length = os.getenv("VISUPATH_CAMERA_FOCAL_LENGTH_PX")
        self.depth_tracker = MonocularDepthTracker(
            horizontal_fov_deg=float(os.getenv("VISUPATH_CAMERA_HORIZONTAL_FOV_DEG", "62")),
            focal_length_px=float(focal_length) if focal_length else None,
            max_match_distance_ratio=float(os.getenv("VISUPATH_TRACK_MATCH_DISTANCE_RATIO", "0.18")),
            smoothing_alpha=float(os.getenv("VISUPATH_SPEED_SMOOTHING_ALPHA", "0.45")),
            stale_after_s=float(os.getenv("VISUPATH_TRACK_STALE_AFTER", "2.0")),
        )
        self.camera_index = int(os.getenv("VISUPATH_CAMERA_INDEX", "0"))
        self.alert_interval = float(os.getenv("VISUPATH_ALERT_INTERVAL", "1.5"))
        self.depth_alert_min_m = float(os.getenv("VISUPATH_DEPTH_ALERT_MIN_M", "1.0"))
        self.last_alert = 0.0

    def alerts(self):
        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")

        print(f"[VisuPath] Object detection running with {self.detector.model_path}")
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    time.sleep(0.1)
                    continue

                detections = self.detector.detect(frame)
                estimates = self.depth_tracker.estimate(detections, frame.shape)
                alert = self._build_alert(estimates, frame.shape)
                if alert and time.monotonic() - self.last_alert >= self.alert_interval:
                    self.last_alert = time.monotonic()
                    yield alert
        finally:
            camera.release()

    def _build_alert(self, estimates: list[DepthEstimate], frame_shape: tuple[int, int, int]) -> Optional[str]:
        if not estimates:
            return None

        _, frame_width = frame_shape[:2]
        measurable = [
            estimate
            for estimate in estimates
            if estimate.distance_m is None or estimate.distance_m >= self.depth_alert_min_m
        ]
        candidates = measurable or estimates
        priority = sorted(
            candidates,
            key=lambda estimate: (
                estimate.distance_m if estimate.distance_m is not None else float("inf"),
                -estimate.area,
            ),
        )[0]
        direction = self._direction(priority.center_x, frame_width)
        distance = self._distance_hint(priority.distance_m)
        motion = self._motion_hint(priority.radial_velocity_mps)
        confidence = round(priority.confidence * 100)
        return f"{priority.label} detected {direction}, {distance}, {motion} ({confidence}% confidence)"

    @staticmethod
    def _direction(center_x: float, frame_width: int) -> str:
        third = frame_width / 3
        if center_x < third:
            return "to your left"
        if center_x > third * 2:
            return "to your right"
        return "ahead"

    @staticmethod
    def _distance_hint(distance_m: Optional[float]) -> str:
        if distance_m is None:
            return "distance unknown"
        if distance_m < 1.0:
            return "within 1 meter"
        if distance_m < 10.0:
            return f"{distance_m:.1f} meters away"
        return f"{round(distance_m)} meters away"

    @staticmethod
    def _motion_hint(radial_velocity_mps: Optional[float]) -> str:
        if radial_velocity_mps is None or abs(radial_velocity_mps) < 0.15:
            return "holding steady"
        speed = abs(radial_velocity_mps)
        if radial_velocity_mps < 0:
            return f"approaching at {speed:.1f} meters per second"
        return f"departing at {speed:.1f} meters per second"
