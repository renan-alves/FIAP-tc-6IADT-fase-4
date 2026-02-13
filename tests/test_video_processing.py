"""
Testes unitários para análise de vídeo.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.video_processing.video_analyzer import VideoAnalyzer, VideoMetadata
from src.detection.yolo_detector import YOLODetector
from config.logger import get_logger

logger = get_logger(__name__)


class TestVideoMetadata:
    """Testes para VideoMetadata."""
    
    def test_video_metadata_creation(self):
        """Testar criação de VideoMetadata."""
        metadata = VideoMetadata(
            filename="test.mp4",
            frame_count=300,
            fps=30,
            width=640,
            height=480,
            duration_seconds=10.0,
            codec="h264"
        )
        
        assert metadata.filename == "test.mp4"
        assert metadata.frame_count == 300
        assert metadata.fps == 30
        assert metadata.duration_seconds == 10.0
    
    def test_video_metadata_to_dict(self):
        """Testar conversão para dicionário."""
        metadata = VideoMetadata(
            filename="test.mp4",
            frame_count=300,
            fps=30,
            width=640,
            height=480,
            duration_seconds=10.0,
            codec="h264"
        )
        
        metadata_dict = metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['filename'] == "test.mp4"
        assert 'duration_formatted' in metadata_dict


class TestVideoAnalyzer:
    """Testes para VideoAnalyzer."""
    
    @pytest.fixture
    def detector(self):
        """Criar detector mock."""
        return Mock(spec=YOLODetector)
    
    @pytest.fixture
    def analyzer(self, detector):
        """Criar analisador de vídeo."""
        return VideoAnalyzer(detector=detector)
    
    def test_analyzer_initialization(self, analyzer):
        """Testar inicialização do analisador."""
        assert analyzer is not None
        assert analyzer.skip_frames > 0
    
    def test_extract_frames_list(self, analyzer, detector):
        """Testar extração de lista de frames."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
        
        result = analyzer.analyzer.detect_batch(frames)
        
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
