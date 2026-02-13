import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.yolo_detector import YOLODetector
from src.video_processing.video_analyzer import VideoAnalyzer
from src.integration.multi_modal import MultiModalAnalyzer
from src.reports.report_generator import ReportGenerator
from config.logger import get_logger

logger = get_logger(__name__)


def analyze_surgical_video(video_path: str) -> None:
    try:
        detector = YOLODetector()
        analyzer = VideoAnalyzer(detector=detector, skip_frames=5)
        analysis_result = analyzer.analyze_video(video_path, show_progress=True)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def multimodal_analysis_example(video_path: str, 
                                audio_path: str = None,
                                document_path: str = None) -> None:
    logger.info("=" * 80)
    logger.info("EXEMPLO: Análise Multimodal Integrada")
    logger.info("=" * 80)
    
    try:
        multi_analyzer = MultiModalAnalyzer(
            video_path=video_path,
            audio_path=audio_path,
            document_path=document_path
        )
        result = multi_analyzer.analyze()
        
        logger.info("Gerando relatório abrangente...")
        report_paths = multi_analyzer.generate_comprehensive_report(result)
        logger.info("\n" + "=" * 80)
        logger.info("AVALIAÇÃO INTEGRADA")
        logger.info("=" * 80)
        logger.info(f"Modalidades processadas: {result.integrated_assessment.get('modalities_processed', [])}")
        logger.info(f"Nível de risco: {result.risk_level}")
        logger.info(f"Confiança geral: {result.integrated_assessment.get('overall_confidence', 0):.2%}")
        logger.info("\nRecomendações clínicas:")
        for rec in result.clinical_recommendations:
            logger.info(f"  • {rec}")
        
    except Exception as e:
        logger.error(f"Erro na análise multimodal: {e}", exc_info=True)


def test_module_loading() -> None:
    logger.info("=" * 80)
    logger.info("TESTE: Carregamento de Módulos")
    logger.info("=" * 80)
    
    try:
        logger.info("\n✓ Importando módulos...")
        
        from src.detection.yolo_detector import YOLODetector
        logger.info("  ✓ YOLODetector")
        
        from src.video_processing.video_analyzer import VideoAnalyzer
        logger.info("  ✓ VideoAnalyzer")
        
        from src.audio_processing.audio_analyzer import AudioAnalyzer
        logger.info("  ✓ AudioAnalyzer (Fase 2 - estrutura pronta)")
        
        from src.text_processing.text_analyzer import TextAnalyzer
        logger.info("  ✓ TextAnalyzer (Fase 3 - estrutura pronta)")
        
        from src.integration.multi_modal import MultiModalAnalyzer
        logger.info("  ✓ MultiModalAnalyzer")
        
        from src.reports.report_generator import ReportGenerator
        logger.info("  ✓ ReportGenerator")
        
        logger.info("\nTodos os módulos carregados com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro ao carregar módulo: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Assistente especializado em Saúde Feminina"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Executar teste de carregamento de módulos"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Caminho do vídeo a analisar"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Caminho do áudio a analisar (opcional)"
    )
    parser.add_argument(
        "--document",
        type=str,
        help="Caminho do documento a analisar (opcional)"
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Executar análise multimodal"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_module_loading()
    elif args.video:
        if args.multimodal:
            multimodal_analysis_example(args.video, args.audio, args.document)
        else:
            analyze_surgical_video(args.video)
    else:
        test_module_loading()
        logger.info("\n" + "=" * 80)
        logger.info("Exemplos de uso:")
        logger.info("=" * 80)
        logger.info(f"  python {__file__} --test")
        logger.info(f"  python {__file__} --video <path/to/video.mp4>")
        logger.info(f"  python {__file__} --video <path> --audio <path> --document <path> --multimodal")
