"""
Teste de integração - Validar todos os módulos funcionam juntos.
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.yolo_detector import YOLODetector
from src.video_processing.video_analyzer import VideoAnalyzer
from src.integration.multi_modal import MultiModalAnalyzer
from src.reports.report_generator import ReportGenerator
from config.logger import get_logger

logger = get_logger(__name__)


def test_module_imports():
    """Teste: Importar todos os módulos."""
    logger.info("Testando importação de módulos...")
    
    from src.detection.yolo_detector import YOLODetector, DetectionResult
    from src.video_processing.video_analyzer import VideoAnalyzer, VideoAnalysisResult
    from src.audio_processing.audio_analyzer import AudioAnalyzer, AudioAnalysisResult
    from src.text_processing.text_analyzer import TextAnalyzer, TextAnalysisResult
    from src.integration.multi_modal import MultiModalAnalyzer, MultiModalAnalysisResult
    from src.reports.report_generator import ReportGenerator
    
    logger.info("✓ Todos os módulos importados com sucesso")


def test_detector_creation():
    """Teste: Criar detector YOLOv8."""
    logger.info("Testando criação do detector...")
    
    detector = YOLODetector()
    assert detector is not None
    assert detector.model is not None
    
    info = detector.get_model_info()
    assert info['model_type'] == 'YOLOv8'
    
    logger.info("✓ Detector criado com sucesso")


def test_video_analyzer_creation():
    """Teste: Criar analisador de vídeo."""
    logger.info("Testando criação do analisador de vídeo...")
    
    detector = YOLODetector()
    analyzer = VideoAnalyzer(detector=detector)
    
    assert analyzer is not None
    assert analyzer.detector is not None
    
    logger.info("✓ Analisador de vídeo criado com sucesso")


def test_multimodal_analyzer_creation():
    """Teste: Criar analisador multimodal."""
    logger.info("Testando criação do analisador multimodal...")
    
    analyzer = MultiModalAnalyzer()
    
    assert analyzer is not None
    assert analyzer.yolo_detector is not None
    assert analyzer.audio_analyzer is not None
    assert analyzer.text_analyzer is not None
    assert analyzer.report_generator is not None
    
    logger.info("✓ Analisador multimodal criado com sucesso")


def test_report_generator_creation():
    """Teste: Criar gerador de relatórios."""
    logger.info("Testando criação do gerador de relatórios...")
    
    generator_pt = ReportGenerator(language='pt-BR')
    generator_en = ReportGenerator(language='en-US')
    
    assert generator_pt is not None
    assert generator_en is not None
    
    assert 'RELATÓRIO' in generator_pt.context['title']
    assert 'REPORT' in generator_en.context['title']
    
    logger.info("✓ Geradores de relatório criados com sucesso")


def run_all_tests():
    """Executar todos os testes de integração."""
    logger.info("=" * 80)
    logger.info("TESTES DE INTEGRAÇÃO")
    logger.info("=" * 80)
    
    tests = [
        test_module_imports,
        test_detector_creation,
        test_video_analyzer_creation,
        test_multimodal_analyzer_creation,
        test_report_generator_creation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            logger.info("")
            test()
            passed += 1
        except Exception as e:
            logger.error(f"✗ FALHOU: {e}", exc_info=True)
            failed += 1
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Resultados: {passed} passou, {failed} falhou")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_all_tests()
