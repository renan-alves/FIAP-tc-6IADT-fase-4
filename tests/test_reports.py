"""
Testes para geração de relatórios.
"""

import pytest
from datetime import datetime

from src.reports.report_generator import ReportGenerator
from src.video_processing.video_analyzer import VideoMetadata, VideoAnalysisResult
from src.detection.yolo_detector import DetectionResult
from config.logger import get_logger

logger = get_logger(__name__)


class TestReportGenerator:
    """Testes para ReportGenerator."""
    
    @pytest.fixture
    def generator_pt(self):
        """Criar gerador em português."""
        return ReportGenerator(language='pt-BR')
    
    @pytest.fixture
    def generator_en(self):
        """Criar gerador em inglês."""
        return ReportGenerator(language='en-US')
    
    @pytest.fixture
    def sample_analysis(self):
        """Criar análise de exemplo."""
        metadata = VideoMetadata(
            filename="test_surgery.mp4",
            frame_count=1800,
            fps=30,
            width=1920,
            height=1080,
            duration_seconds=60.0,
            codec="h264"
        )
        
        detections = {
            0: [
                DetectionResult(
                    class_id=0,
                    class_name="speculum",
                    confidence=0.95,
                    bbox=(100, 100, 300, 300),
                    frame_id=0
                )
            ]
        }
        
        return VideoAnalysisResult(
            metadata=metadata,
            total_detections=1,
            frames_with_detections=1,
            detection_frames=detections,
            instruments_detected={"speculum": 1},
            frames_analyzed=60
        )
    
    def test_generator_initialization_pt(self, generator_pt):
        """Testar inicialização do gerador português."""
        assert generator_pt is not None
        assert generator_pt.language == 'pt-BR'
    
    def test_generator_initialization_en(self, generator_en):
        """Testar inicialização do gerador inglês."""
        assert generator_en is not None
        assert generator_en.language == 'en-US'
    
    def test_detect_complications_empty(self, generator_pt, sample_analysis):
        """Testar detecção de complicações (nenhuma)."""
        complications = generator_pt._detect_complications(sample_analysis)
        
        assert isinstance(complications, list)
    
    def test_generate_json_report(self, generator_pt, sample_analysis):
        """Testar geração de relatório JSON."""
        report = generator_pt.generate_json_report(sample_analysis)
        
        assert isinstance(report, dict)
        assert 'metadata' in report
        assert 'surgery_info' in report
        assert 'analysis' in report
        assert 'clinical_assessment' in report
    
    def test_generate_text_report(self, generator_pt, sample_analysis):
        """Testar geração de relatório em texto."""
        report_text = generator_pt.generate_text_report(sample_analysis)
        
        assert isinstance(report_text, str)
        assert 'RELATÓRIO' in report_text
        assert 'INSTRUMENTOS' in report_text
    
    def test_text_report_language_pt(self, generator_pt, sample_analysis):
        """Testar textos em português no relatório."""
        report_text = generator_pt.generate_text_report(sample_analysis)
        
        assert 'CIRURGIA' in report_text
        assert 'INSTRUMENTOS UTILIZADOS' in report_text
    
    def test_text_report_language_en(self, generator_en, sample_analysis):
        """Testar textos em inglês no relatório."""
        report_text = generator_en.generate_text_report(sample_analysis)
        
        assert 'SURGERY' in report_text
        assert 'INSTRUMENTS' in report_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
