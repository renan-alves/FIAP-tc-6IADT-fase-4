import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from ultralytics import YOLO
from config.settings import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    DEVICE,
    MEDICAL_INSTRUMENTS,
)
from config.logger import get_logger

logger = get_logger(__name__)


class DetectionResult:
    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        frame_id: int,
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.frame_id = frame_id

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": tuple(float(x) for x in self.bbox),
            "frame_id": self.frame_id,
            "instrument_pt": MEDICAL_INSTRUMENTS.get(self.class_name, self.class_name),
        }


class YOLODetector:
    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        device: str = DEVICE,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.using_default_model = False

        if not Path(model_path).exists():
            logger.warning(f"Modelo não encontrado. Usando YOLOv8n padrão")
            self.model = YOLO("yolov8n.pt")
            self.using_default_model = True
        else:
            self.model = YOLO(model_path)
            self.using_default_model = False

        try:
            self.model.to(device)
        except (AssertionError, RuntimeError):
            self.device = "cpu"
            self.model.to("cpu")
            logger.warning(f"GPU não disponível. Usando CPU")

    def detect_frame(
        self, frame: np.ndarray, frame_id: int = 0
    ) -> List[DetectionResult]:
        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            detections = []

            relevant_coco_classes = (
                {"scissors", "knife", "fork", "spoon", "bottle", "cup"}
                if self.using_default_model
                else None
            )

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        confidence = float(box.conf[0])

                        if self.using_default_model and relevant_coco_classes:
                            if class_name not in relevant_coco_classes:
                                continue

                        bbox = tuple(float(x) for x in box.xyxy[0])

                        detection = DetectionResult(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            frame_id=frame_id,
                        )
                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Erro ao detectar frame {frame_id}: {e}")
            return []

    def detect_batch(
        self, frames: List[np.ndarray], start_frame_id: int = 0
    ) -> Dict[int, List[DetectionResult]]:
        """
        Detectar instrumentos em múltiplos frames.

        Args:
            frames: Lista de frames
            start_frame_id: ID inicial dos frames

        Returns:
            Dicionário mapeando frame_id para lista de DetectionResult
        """
        batch_results = {}

        for idx, frame in enumerate(frames):
            frame_id = start_frame_id + idx
            detections = self.detect_frame(frame, frame_id)
            if detections:
                batch_results[frame_id] = detections

        return batch_results

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        show_confidence: bool = True,
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            instrument_pt = MEDICAL_INSTRUMENTS.get(det.class_name, det.class_name)
            if show_confidence:
                label = f"{instrument_pt} ({det.confidence:.2f})"
            else:
                label = instrument_pt

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - 25),
                (x1 + text_size[0] + 5, y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return annotated_frame

    def get_model_info(self) -> Dict:
        """
        Retorna informações básicas sobre o modelo YOLO carregado.

        Returns:
            Dicionário com informações do modelo (class, device, model_path, etc.)
        """
        model = getattr(self, "model", None)
        if model is None:
            return {}

        try:
            info: Dict = {
                "class": model.__class__.__name__,
                "device": getattr(model, "device", None),
            }

            # Alguns wrappers de modelo expõem caminho/weights; coletar quando disponível
            if hasattr(model, "model_path"):
                info["model_path"] = getattr(model, "model_path")
            elif hasattr(model, "weights"):
                info["weights"] = getattr(model, "weights")
            elif hasattr(model, "cfg"):
                info["cfg"] = getattr(model, "cfg")

            return info
        except Exception:
            # Não propagar erro de introspecção do modelo; retornar dado mínimo
            return {"class": model.__class__.__name__}
