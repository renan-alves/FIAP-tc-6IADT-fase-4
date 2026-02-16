"""
Geração de relatórios automáticos especializados em cirurgias ginecológicas.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from src.video_processing.video_analyzer import VideoAnalysisResult
from config.logger import get_logger
from config.settings import MEDICAL_INSTRUMENTS, REPORT_LANGUAGE

logger = get_logger(__name__)


@dataclass
class SurgicalSummary:
    """Resumo da cirurgia."""
    surgery_date: str
    duration_minutes: float
    success: bool
    complications: List[str]
    instruments_used: List[str]
    observations: str


class ReportGenerator:
    """Gerador de relatórios cirúrgicos especializados."""
    
    def __init__(self, language: str = REPORT_LANGUAGE):
        """
        Inicializar gerador de relatórios.
        
        Args:
            language: Idioma do relatório (pt-BR, en-US)
        """
        self.language = language
        self.context = self._load_context()
        logger.info(f"ReportGenerator inicializado com idioma: {language}")
    
    def _load_context(self) -> Dict:
        """Carregar contexto baseado no idioma."""
        context_pt = {
            'title': 'RELATÓRIO DE CIRURGIA GINECOLÓGICA',
            'surgery_date': 'Data da Cirurgia:',
            'duration': 'Duração do Procedimento:',
            'instruments_section': 'INSTRUMENTOS UTILIZADOS',
            'analysis_section': 'ANÁLISE PRELIMINAR DA DETECÇÃO',
            'summary_section': 'RESUMO DA CIRURGIA',
            'frames_analyzed': 'Frames Analisados:',
            'total_detections': 'Total de Detecções:',
            'success_label': 'SUCESSO CIRÚRGICO',
            'complications_label': 'COMPLICAÇÕES DETECTADAS',
            'no_complications': 'Nenhuma complicação identificada',
            'recommendations': 'RECOMENDAÇÕES CLÍNICAS',
            'generated': 'Relatório gerado em:',
        }
        
        context_en = {
            'title': 'GYNECOLOGICAL SURGERY REPORT',
            'surgery_date': 'Surgery Date:',
            'duration': 'Procedure Duration:',
            'instruments_section': 'INSTRUMENTS USED',
            'analysis_section': 'PRELIMINARY DETECTION ANALYSIS',
            'summary_section': 'SURGERY SUMMARY',
            'frames_analyzed': 'Frames Analyzed:',
            'total_detections': 'Total Detections:',
            'success_label': 'SURGICAL SUCCESS',
            'complications_label': 'DETECTED COMPLICATIONS',
            'no_complications': 'No complications identified',
            'recommendations': 'CLINICAL RECOMMENDATIONS',
            'generated': 'Report generated at:',
        }
        
        return context_pt if self.language == 'pt-BR' else context_en
    
    def _detect_complications(self, analysis: VideoAnalysisResult) -> List[str]:
        """
        Detectar possíveis complicações baseado na análise de vídeo.
        
        Args:
            analysis: Resultado da análise de vídeo
            
        Returns:
            Lista de complicações detectadas
        """
        complications = []
        
        # Lógica simples de detecção de complicações
        # Estas são heurísticas que podem ser melhoradas com ML
        
        # Se houver muita variação de instrumentos pode indicar mudança de técnica
        if len(analysis.instruments_detected) > 5:
            complications.append(
                "Múltiplas mudanças de instrumentos detectadas - possível ajuste intraoperatório"
            )
        
        # Se houver poucas detecções em vídeo longo
        if (analysis.metadata.duration_seconds > 60 and 
            analysis.total_detections < 10):
            complications.append(
                "Poucas detecções registradas - verificar qualidade do vídeo"
            )
        
        # Se houver muita atividade no final (last 20% do vídeo)
        if analysis.detection_frames:
            last_frame = max(analysis.detection_frames.keys())
            threshold_frame = analysis.metadata.frame_count * 0.8
            late_detections = [f for f in analysis.detection_frames.keys() 
                              if f > threshold_frame]
            
            if len(late_detections) > len(analysis.detection_frames) * 0.3:
                complications.append(
                    "Atividade elevada nas fases finais - possível complicação/ajuste"
                )
        
        return complications
    
    def generate_json_report(self, analysis: VideoAnalysisResult, 
                            output_path: str = None) -> Dict:
        """
        Gerar relatório em JSON.
        
        Args:
            analysis: Resultado da análise de vídeo
            output_path: Caminho de saída (opcional)
            
        Returns:
            Dicionário com relatório
        """
        logger.info("Gerando relatório em formato JSON")
        
        complications = self._detect_complications(analysis)
        success = len(complications) == 0
        
        # Traduzir nomes de instrumentos
        instruments_pt = []
        for instrument in analysis.instruments_detected.keys():
            pt_name = MEDICAL_INSTRUMENTS.get(instrument, instrument)
            instruments_pt.append({
                'name': pt_name,
                'count': analysis.instruments_detected[instrument]
            })
        
        report = {
            'metadata': {
                'report_generated': datetime.now().isoformat(),
                'language': self.language,
                'version': '1.0'
            },
            'surgery_info': {
                'date': datetime.now().strftime('%d/%m/%Y'),
                'filename': analysis.metadata.filename,
                'duration_seconds': analysis.metadata.duration_seconds,
                'duration_minutes': analysis.metadata.duration_seconds / 60,
                'video_resolution': f"{analysis.metadata.width}x{analysis.metadata.height}",
            },
            'analysis': {
                'frames_analyzed': analysis.frames_analyzed,
                'frames_with_detections': analysis.frames_with_detections,
                'total_detections': analysis.total_detections,
                'instruments_detected': instruments_pt,
            },
            'clinical_assessment': {
                'success': success,
                'complications': complications if complications else [],
                'risk_level': 'LOW' if success else 'MEDIUM' if len(complications) == 1 else 'HIGH'
            },
            'recommendations': self._generate_recommendations(success, complications)
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Relatório JSON salvo em: {output_path}")
        
        return report
    
    def generate_text_report(self, analysis: VideoAnalysisResult,
                            complications: List[str] = None,
                            output_path: str = None) -> str:
        """
        Gerar relatório em texto (formato texto estruturado).
        
        Args:
            analysis: Resultado da análise de vídeo
            complications: Lista de complicações
            output_path: Caminho de saída (opcional)
            
        Returns:
            Texto do relatório
        """
        logger.info("Gerando relatório em formato texto")
        
        if complications is None:
            complications = self._detect_complications(analysis)
        
        success = len(complications) == 0
        ctx = self.context
        
        # Detectar se está usando modelo COCO (classes não médicas)
        coco_classes = {
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        }
        
        detected_classes = set(analysis.instruments_detected.keys()) if analysis.instruments_detected else set()
        using_default_model = bool(detected_classes & coco_classes)
        
        lines = [
            "=" * 80,
            ctx['title'],
            "=" * 80,
            ""
        ]
        
        if using_default_model:
            lines.extend([
                "⚠️  AVISO IMPORTANTE - MODELO NÃO TREINADO ⚠️",
                "-" * 80,
                "Este relatório foi gerado usando o modelo YOLOv8 GENÉRICO (COCO dataset).",
                "As 'detecções' abaixo NÃO são de instrumentos ginecológicos reais!",
                "",
                "O modelo está identificando objetos aleatórios (frutas, animais, objetos",
                "domésticos) porque não foi treinado com imagens de cirurgias.",
                "",
                "⚡ PARA CORRIGIR:",
                "   1. Treinar modelo com dataset de instrumentos ginecológicos",
                "   2. Executar: python prepare_model.py",
                "   3. Ver guia: cat QUICKSTART_MODEL.md",
                "   4. Comparar modelos: python compare_models.py",
                "",
                "Este relatório serve apenas como DEMONSTRAÇÃO da infraestrutura.",
                "=" * 80,
                ""
            ])
        
        lines.extend([
            f"Arquivo: {analysis.metadata.filename}",
            f"{ctx['surgery_date']} {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            f"{ctx['duration']} {analysis.metadata.duration_seconds / 60:.2f} minutos",
            "",
            "-" * 80,
            ctx['instruments_section'],
            "-" * 80,
        ])
        
        if analysis.instruments_detected:
            for instrument, count in sorted(analysis.instruments_detected.items(),
                                           key=lambda x: x[1], reverse=True):
                pt_name = MEDICAL_INSTRUMENTS.get(instrument, instrument)
                lines.append(f"  • {pt_name}: {count} detecções")
        else:
            lines.append("  Nenhum instrumento detectado")
        
        lines.extend([
            "",
            "-" * 80,
            ctx['analysis_section'],
            "-" * 80,
            f"{ctx['frames_analyzed']} {analysis.frames_analyzed}",
            f"{ctx['total_detections']} {analysis.total_detections}",
            f"Frames com detecções: {analysis.frames_with_detections}",
            "",
            "-" * 80,
            ctx['summary_section'],
            "-" * 80,
        ])
        
        if success:
            lines.append(f"✓ {ctx['success_label']}")
        else:
            lines.append(f"⚠ {ctx['complications_label']}:")
            for complication in complications:
                lines.append(f"  • {complication}")
        
        lines.extend([
            "",
            "-" * 80,
            ctx['recommendations'],
            "-" * 80,
        ])
        
        recommendations = self._generate_recommendations(success, complications)
        for rec in recommendations:
            lines.append(f"  • {rec}")
        
        lines.extend([
            "",
            f"{ctx['generated']} {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            "=" * 80,
        ])
        
        report_text = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Relatório TXT salvo em: {output_path}")
        
        return report_text
    
    def _generate_recommendations(self, success: bool, 
                                 complications: List[str]) -> List[str]:
        """Gerar recomendações clínicas."""
        recommendations = []
        
        if success:
            recommendations.append("Procedimento realizado com sucesso - acompanhamento pós-operatório padrão")
            recommendations.append("Agendar consulta de seguimento em 7 dias")
        else:
            if len(complications) > 0:
                recommendations.append("Acompanhamento clínico reforçado recomendado")
                recommendations.append("Realizar exame de acompanhamento adicional")
            recommendations.append("Documentar todas as observações no prontuário da paciente")
        
        recommendations.append("Análise completada por assistente com IA - revisar com especialista")
        
        return recommendations
    
    def generate_all_formats(self, analysis: VideoAnalysisResult,
                            output_dir: str = "data/reports") -> Dict[str, str]:
        """
        Gerar relatório em todos os formatos.
        
        Args:
            analysis: Resultado da análise de vídeo
            output_dir: Diretório de saída
            
        Returns:
            Dicionário com paths dos relatórios
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(analysis.metadata.filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        paths = {}
        
        # JSON
        json_path = output_dir / f"{base_name}_report_{timestamp}.json"
        self.generate_json_report(analysis, str(json_path))
        paths['json'] = str(json_path)
        
        # Texto
        txt_path = output_dir / f"{base_name}_report_{timestamp}.txt"
        self.generate_text_report(analysis, output_path=str(txt_path))
        paths['txt'] = str(txt_path)
        
        logger.info(f"Relatórios gerados em: {output_dir}")
        
        return paths
