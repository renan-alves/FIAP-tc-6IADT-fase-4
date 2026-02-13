"""
Assistente especializado em Saúde Feminina com IA integrada.
"""

__version__ = "0.1.0"
__author__ = "FIAP Tech Challenge"

from src.detection.yolo_detector import YOLODetector, DetectionResult
from src.video_processing.video_analyzer import VideoAnalyzer, VideoAnalysisResult
from src.audio_processing.audio_analyzer import AudioAnalyzer
from src.text_processing.text_analyzer import TextAnalyzer
from src.integration.multi_modal import MultiModalAnalyzer, MultiModalAnalysisResult
from src.reports.report_generator import ReportGenerator

__all__ = [
    'YOLODetector',
    'DetectionResult',
    'VideoAnalyzer',
    'VideoAnalysisResult',
    'AudioAnalyzer',
    'TextAnalyzer',
    'MultiModalAnalyzer',
    'MultiModalAnalysisResult',
    'ReportGenerator',
]
