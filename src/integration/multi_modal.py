"""
Integração multimodal de dados médicos.
Combina análise de vídeo, áudio e texto para decisão clínica.
"""

from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime

from src.detection.yolo_detector import YOLODetector
from src.video_processing.video_analyzer import VideoAnalyzer, VideoAnalysisResult
from src.audio_processing.audio_analyzer import AudioAnalyzer, AudioAnalysisResult
from src.text_processing.text_analyzer import TextAnalyzer, TextAnalysisResult
from src.reports.report_generator import ReportGenerator
from config.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MultiModalAnalysisResult:
    """Resultado da análise multimodal completa."""
    timestamp: str
    video_analysis: Optional[VideoAnalysisResult] = None
    audio_analysis: Optional[AudioAnalysisResult] = None
    text_analysis: Optional[TextAnalysisResult] = None
    integrated_assessment: Dict = field(default_factory=dict)
    clinical_recommendations: List[str] = field(default_factory=list)
    risk_level: str = "UNKNOWN"
    
    def to_dict(self) -> Dict:
        """Converter resultado para dicionário."""
        return {
            'timestamp': self.timestamp,
            'video_analysis': self.video_analysis.to_dict() if self.video_analysis else None,
            'audio_analysis': self.audio_analysis.__dict__ if self.audio_analysis else None,
            'text_analysis': self.text_analysis.__dict__ if self.text_analysis else None,
            'integrated_assessment': self.integrated_assessment,
            'clinical_recommendations': self.clinical_recommendations,
            'risk_level': self.risk_level
        }


class MultiModalAnalyzer:
    """
    Analisador multimodal integrado.
    Combina análise de vídeo, áudio e texto para suporte a decisão clínica.
    """
    
    def __init__(self, 
                 yolo_detector: Optional[YOLODetector] = None,
                 audio_analyzer: Optional[AudioAnalyzer] = None,
                 text_analyzer: Optional[TextAnalyzer] = None,
                 report_generator: Optional[ReportGenerator] = None):
        """
        Inicializar analisador multimodal.
        
        Args:
            yolo_detector: Detector YOLOv8 (criado se não fornecido)
            audio_analyzer: Analisador de áudio
            text_analyzer: Analisador de texto
            report_generator: Gerador de relatórios
        """
        self.yolo_detector = yolo_detector or YOLODetector()
        self.audio_analyzer = audio_analyzer or AudioAnalyzer()
        self.text_analyzer = text_analyzer or TextAnalyzer()
        self.report_generator = report_generator or ReportGenerator()
        
        logger.info("MultiModalAnalyzer inicializado")
        logger.info("Módulos ativos: Vídeo (✓), Áudio (planejado), Texto (planejado)")
    
    def analyze(self, 
                video_path: Optional[str] = None,
                audio_path: Optional[str] = None,
                document_path: Optional[str] = None) -> MultiModalAnalysisResult:
        """
        Executar análise multimodal completa.
        
        Args:
            video_path: Caminho do vídeo cirúrgico
            audio_path: Caminho do áudio (futuro)
            document_path: Caminho do documento médico (futuro)
            
        Returns:
            MultiModalAnalysisResult com análise integrada
        """
        logger.info("Iniciando análise multimodal")
        
        result = MultiModalAnalysisResult(
            timestamp=datetime.now().isoformat()
        )
        
        # Análise de vídeo
        if video_path and Path(video_path).exists():
            logger.info("Processando análise de vídeo...")
            video_analyzer = VideoAnalyzer(self.yolo_detector)
            result.video_analysis = video_analyzer.analyze_video(video_path)
        
        # Análise de áudio (Fase 2)
        if audio_path and Path(audio_path).exists():
            logger.info("[FASE 2 - NÃO IMPLEMENTADO] Processando análise de áudio...")
            result.audio_analysis = self.audio_analyzer.analyze_audio(audio_path)
        
        # Análise de texto (Fase 3)
        if document_path and Path(document_path).exists():
            logger.info("[FASE 3 - NÃO IMPLEMENTADO] Processando análise de documento...")
            result.text_analysis = self.text_analyzer.analyze_document(document_path)
        
        # Integração e avaliação clínica
        self._integrate_analysis(result)
        
        logger.info("Análise multimodal concluída")
        return result
    
    def _integrate_analysis(self, multimodal_result: MultiModalAnalysisResult) -> None:
        """
        Integrar análises de diferentes modalidades.
        
        Args:
            multimodal_result: Resultado a ser integrado
        """
        logger.info("Integrando análises multimodais...")
        
        assessment = {
            'modalities_processed': [],
            'data_concordance': 'UNKNOWN',
            'overall_confidence': 0.0
        }
        
        # Processar vídeo se disponível
        if multimodal_result.video_analysis:
            assessment['modalities_processed'].append('vídeo')
            assessment['video_quality'] = 'OK'
            assessment['instruments_detected'] = len(
                multimodal_result.video_analysis.instruments_detected
            )
        
        # Processar áudio se disponível (Fase 2)
        if multimodal_result.audio_analysis:
            assessment['modalities_processed'].append('áudio')
            assessment['audio_quality'] = 'UNKNOWN'
        
        # Processar texto se disponível (Fase 3)
        if multimodal_result.text_analysis:
            assessment['modalities_processed'].append('texto')
            assessment['document_quality'] = 'UNKNOWN'
        
        # Calcular confiança geral
        confidence = 1.0 / len(assessment['modalities_processed']) \
            if assessment['modalities_processed'] else 0.0
        assessment['overall_confidence'] = min(confidence, 1.0)
        
        multimodal_result.integrated_assessment = assessment
        
        # Gerar recomendações
        self._generate_clinical_recommendations(multimodal_result)
        
        # Determinar nível de risco
        self._assess_risk_level(multimodal_result)
    
    def _generate_clinical_recommendations(self, 
                                          multimodal_result: MultiModalAnalysisResult) -> None:
        """
        Gerar recomendações clínicas baseadas na análise integrada.
        
        Args:
            multimodal_result: Resultado multimodal
        """
        recommendations = []
        
        if multimodal_result.video_analysis:
            # Recomendações baseadas em vídeo
            if multimodal_result.video_analysis.total_detections == 0:
                recommendations.append(
                    "Qualidade do vídeo pode estar baixa - verificar gravação"
                )
            elif multimodal_result.video_analysis.total_detections > 50:
                recommendations.append(
                    "Alto número de detecções - possível mudança frequente de instrumentos"
                )
        
        if len(multimodal_result.integrated_assessment.get('modalities_processed', [])) > 1:
            recommendations.append(
                "Análise multimodal concluída - recomenda-se revisão por especialista"
            )
        
        recommendations.append(
            "Este assistente é uma ferramenta de suporte e não substitui avaliação clínica"
        )
        
        multimodal_result.clinical_recommendations = recommendations
    
    def _assess_risk_level(self, multimodal_result: MultiModalAnalysisResult) -> None:
        """
        Avaliar nível de risco geral.
        
        Args:
            multimodal_result: Resultado multimodal
        """
        risk_level = "LOW"
        
        if multimodal_result.video_analysis:
            # Lógica simples de risco
            total_dets = multimodal_result.video_analysis.total_detections
            if total_dets == 0:
                risk_level = "UNKNOWN"
            elif total_dets > 50:
                risk_level = "MEDIUM"
        
        multimodal_result.risk_level = risk_level
        logger.info(f"Nível de risco avaliado: {risk_level}")
    
    def generate_comprehensive_report(self, 
                                     multimodal_result: MultiModalAnalysisResult,
                                     output_dir: str = "data/reports") -> Dict[str, str]:
        """
        Gerar relatório abrangente com todas as análises.
        
        Args:
            multimodal_result: Resultado multimodal
            output_dir: Diretório de saída
            
        Returns:
            Dicionário com paths dos relatórios
        """
        logger.info("Gerando relatório abrangente...")
        
        paths = {}
        
        if multimodal_result.video_analysis:
            paths.update(
                self.report_generator.generate_all_formats(
                    multimodal_result.video_analysis,
                    output_dir
                )
            )
        
        return paths
