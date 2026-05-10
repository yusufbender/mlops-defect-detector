"""
Defect detection model wrapper.

V1 (current): Uses pretrained YOLOv8n on COCO classes as a placeholder
              to validate the inference pipeline end-to-end.
              Defect score is derived from detection confidence + heuristics.

V2 (planned): Fine-tune on MVTec AD dataset for real defect detection.
V3 (planned): Replace with PaDiM/PatchCore (anomalib) for proper anomaly detection.
"""
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/yolov8n.pt")
DEFAULT_CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))


class DefectDetector:
    """YOLOv8-based detector with anomaly scoring heuristics."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model: Optional[YOLO] = None
        self.version = "yolov8n-coco-v1"

    def load(self) -> None:
        """Load the model. Called once at app startup."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        logger.info("Loading YOLOv8 model from %s", self.model_path)
        self.model = YOLO(self.model_path)
        # Warm-up with a dummy image (first inference is slow)
        dummy = Image.new("RGB", (320, 320), (128, 128, 128))
        self.model.predict(dummy, verbose=False, conf=self.conf_threshold)
        logger.info("Model loaded and warmed up")

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL image, return defect-detection style result.

        Heuristic anomaly score (v1):
          - Run YOLOv8 detection
          - If no objects detected -> low anomaly (looks "uniform")
          - If many low-confidence detections -> high anomaly (cluttered/uncertain)
          - Top-1 confidence is the inverse anomaly score
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        results = self.model.predict(
            image,
            verbose=False,
            conf=self.conf_threshold,
        )

        boxes = results[0].boxes
        num_detections = len(boxes) if boxes is not None else 0

        if num_detections == 0:
            return {
                "defective": False,
                "defect_type": "ok",
                "confidence": 0.85,
                "bbox": None,
                "num_detections": 0,
                "model_version": self.version,
                "_note": "v1 heuristic: no detections",
            }

        confs = boxes.conf.cpu().numpy()
        top_idx = int(np.argmax(confs))
        top_conf = float(confs[top_idx])
        top_class_id = int(boxes.cls[top_idx].item())
        top_class_name = self.model.names[top_class_id]
        top_bbox = boxes.xyxy[top_idx].cpu().numpy().astype(int).tolist()

        is_defective = top_conf < 0.5 or num_detections > 5
        anomaly_score = round(1.0 - top_conf, 3) if is_defective else 0.0

        return {
            "defective": is_defective,
            "defect_type": top_class_name if is_defective else "ok",
            "confidence": round(top_conf, 3),
            "anomaly_score": anomaly_score,
            "bbox": top_bbox if is_defective else None,
            "num_detections": num_detections,
            "model_version": self.version,
            "_note": "v1 heuristic on COCO-pretrained YOLOv8n",
        }


detector = DefectDetector()
