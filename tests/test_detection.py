"""
Testes unitários para o detector YOLOv8.
"""

import pytest
import numpy as np
from pathlib import Path

from src.detection.yolo_detector import YOLODetector, DetectionResult
from config.logger import get_logger

logger = get_logger(__name__)


class TestDetectionResult:
    """Testes para DetectionResult."""

    def test_detection_result_creation(self):
        """Testar criação de DetectionResult."""
        result = DetectionResult(
            class_id=0,
            class_name="speculum",
            confidence=0.95,
            bbox=(10, 20, 100, 150),
            frame_id=0,
        )

        assert result.class_id == 0
        assert result.class_name == "speculum"
        assert result.confidence == 0.95
        assert result.bbox == (10, 20, 100, 150)
        assert result.frame_id == 0

    def test_detection_result_to_dict(self):
        """Testar conversão para dicionário."""
        result = DetectionResult(
            class_id=0,
            class_name="speculum",
            confidence=0.95,
            bbox=(10, 20, 100, 150),
            frame_id=0,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "class_id" in result_dict
        assert "confidence" in result_dict
        assert result_dict["instrument_pt"] == "Espéculo"


class TestYOLODetector:
    """Testes para YOLODetector."""

    @pytest.fixture
    def detector(self):
        """Criar instância do detector."""
        return YOLODetector()

    def test_detector_initialization(self, detector):
        """Testar inicialização do detector."""
        assert detector is not None
        assert detector.model is not None
        assert detector.confidence_threshold > 0

    def test_detect_frame_with_empty_frame(self, detector):
        """Testar detecção em frame vazio."""
        # Criar frame vazio (preto)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_frame(frame, frame_id=0)

        assert isinstance(detections, list)

    def test_detect_batch(self, detector):
        """Testar detecção em batch."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]

        results = detector.detect_batch(frames, start_frame_id=0)

        assert isinstance(results, dict)

    def test_get_model_info(self, detector):
        """Testar obtenção de informações do modelo."""
        info = detector.get_model_info()

        assert isinstance(info, dict)
        assert "device" in info


class TestDetectionResult_DrawAnnotations:
    """Testes para anotação de detecções."""

    def test_draw_detections(self):
        """Testar desenho de detecções."""
        detector = YOLODetector()

        # Criar frame de exemplo
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Criar detecção de exemplo
        detections = [
            DetectionResult(
                class_id=0,
                class_name="speculum",
                confidence=0.95,
                bbox=(50, 50, 200, 200),
                frame_id=0,
            )
        ]

        annotated = detector.draw_detections(frame, detections)

        assert annotated.shape == frame.shape
        # Verificar que algo foi modificado
        assert not np.array_equal(frame, annotated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
