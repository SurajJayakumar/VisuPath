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


class ObjectDetectionPipeline:
    def __init__(self):
        labels_path = Path(os.getenv("VISUPATH_LABELS_PATH", Path(__file__).with_name("coco_labels.txt")))
        self.detector = YoloTfliteDetector(labels_path=labels_path)
        self.camera_index = int(os.getenv("VISUPATH_CAMERA_INDEX", "0"))
        self.alert_interval = float(os.getenv("VISUPATH_ALERT_INTERVAL", "1.5"))
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
                alert = self._build_alert(detections, frame.shape)
                if alert and time.monotonic() - self.last_alert >= self.alert_interval:
                    self.last_alert = time.monotonic()
                    yield alert
        finally:
            camera.release()

    def _build_alert(self, detections: list[Detection], frame_shape: tuple[int, int, int]) -> Optional[str]:
        if not detections:
            return None

        frame_height, frame_width = frame_shape[:2]
        priority = sorted(detections, key=lambda detection: detection.area, reverse=True)[0]
        direction = self._direction(priority.center_x, frame_width)
        distance = self._distance_hint(priority.box, frame_height)
        confidence = round(priority.confidence * 100)
        return f"{priority.label} detected {direction}, {distance} ({confidence}% confidence)"

    @staticmethod
    def _direction(center_x: float, frame_width: int) -> str:
        third = frame_width / 3
        if center_x < third:
            return "to your left"
        if center_x > third * 2:
            return "to your right"
        return "ahead"

    @staticmethod
    def _distance_hint(box: tuple[float, float, float, float], frame_height: int) -> str:
        _, y1, _, y2 = box
        height_ratio = (y2 - y1) / max(frame_height, 1)
        if height_ratio > 0.55:
            return "very close"
        if height_ratio > 0.30:
            return "nearby"
        return "farther away"
