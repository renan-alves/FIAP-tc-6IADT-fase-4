"""
Processamento e análise de vídeos cirúrgicos.
"""

import cv2
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datetime import timedelta
from tqdm import tqdm

from src.detection.yolo_detector import YOLODetector, DetectionResult
from config.settings import VIDEO_SKIP_FRAMES
from config.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoMetadata:
    """Metadados do vídeo."""
    filename: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    codec: str
    
    def to_dict(self) -> Dict:
        """Converter para dicionário."""
        return {
            'filename': self.filename,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'duration_seconds': self.duration_seconds,
            'duration_formatted': str(timedelta(seconds=int(self.duration_seconds))),
            'codec': self.codec
        }


@dataclass
class VideoAnalysisResult:
    """Resultado da análise de vídeo."""
    metadata: VideoMetadata
    total_detections: int
    frames_with_detections: int
    detection_frames: Dict[int, List[DetectionResult]]
    instruments_detected: Dict[str, int]  # {instrument: count}
    frames_analyzed: int
    
    def to_dict(self) -> Dict:
        """Converter para dicionário."""
        return {
            'metadata': self.metadata.to_dict(),
            'total_detections': self.total_detections,
            'frames_with_detections': self.frames_with_detections,
            'frames_analyzed': self.frames_analyzed,
            'instruments_detected': self.instruments_detected,
            'detection_details': {
                str(frame_id): [det.to_dict() for det in dets]
                for frame_id, dets in self.detection_frames.items()
            }
        }


class VideoAnalyzer:
    """Analisador de vídeos cirúrgicos."""
    
    def __init__(self, detector: YOLODetector, 
                 skip_frames: int = VIDEO_SKIP_FRAMES):
        """
        Inicializar analisador de vídeos.
        
        Args:
            detector: Instância do YOLODetector
            skip_frames: Número de frames a pular entre análises
        """
        self.detector = detector
        self.skip_frames = skip_frames
        logger.info(f"VideoAnalyzer inicializado com skip_frames={skip_frames}")
    
    def extract_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extrair metadados do vídeo.
        
        Args:
            video_path: Caminho do vídeo
            
        Returns:
            VideoMetadata
        """
        cap = cv2.VideoCapture(video_path)
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Obter codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            duration_seconds = frame_count / fps if fps > 0 else 0
            
            metadata = VideoMetadata(
                filename=Path(video_path).name,
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                codec=codec
            )
            
            logger.info(f"Metadados extraídos: {frame_count} frames, {fps} fps, "
                       f"{width}x{height}, {duration_seconds:.2f}s")
            
            return metadata
            
        finally:
            cap.release()
    
    def analyze_video(self, video_path: str, 
                     show_progress: bool = True) -> VideoAnalysisResult:
        """
        Analisar vídeo e detectar instrumentos.
        
        Args:
            video_path: Caminho do vídeo
            show_progress: Mostrar barra de progresso
            
        Returns:
            VideoAnalysisResult
        """
        logger.info(f"Iniciando análise de vídeo: {video_path}")
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        metadata = self.extract_metadata(video_path)
        cap = cv2.VideoCapture(video_path)
        
        detection_frames: Dict[int, List[DetectionResult]] = {}
        instruments_count: Dict[str, int] = {}
        total_detections = 0
        frames_with_detections = 0
        frames_analyzed = 0
        
        try:
            frames_to_process = metadata.frame_count // self.skip_frames
            
            pbar = tqdm(total=frames_to_process, 
                       desc="Analisando vídeo",
                       disable=not show_progress)
            
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_id % self.skip_frames == 0:
                    frames_analyzed += 1
                    
                    detections = self.detector.detect_frame(frame, frame_id)
                    
                    if detections:
                        detection_frames[frame_id] = detections
                        frames_with_detections += 1
                        total_detections += len(detections)
                        
                        for det in detections:
                            instruments_count[det.class_name] = \
                                instruments_count.get(det.class_name, 0) + 1
                    
                    pbar.update(1)
                
                frame_id += 1
            
            pbar.close()
            
            result = VideoAnalysisResult(
                metadata=metadata,
                total_detections=total_detections,
                frames_with_detections=frames_with_detections,
                detection_frames=detection_frames,
                instruments_detected=instruments_count,
                frames_analyzed=frames_analyzed
            )
            
            logger.info(f"Análise concluída: {total_detections} detecções em "
                       f"{frames_with_detections} frames de {frames_analyzed} analisados")
            
            return result
            
        finally:
            cap.release()
    
    def extract_frames(self, video_path: str, 
                      output_dir: str = None,
                      skip_frames: int = None) -> List[str]:
        """
        Extrair frames do vídeo.
        
        Args:
            video_path: Caminho do vídeo
            output_dir: Diretório de saída para frames
            skip_frames: Frames a pular (padrão: self.skip_frames)
            
        Returns:
            Lista com paths dos frames salvos
        """
        if skip_frames is None:
            skip_frames = self.skip_frames
        
        if output_dir is None:
            output_dir = Path("data/frames")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extraindo frames de {video_path} para {output_dir}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        saved_frames = []
        frame_id = 0
        
        try:
            pbar = tqdm(total=frame_count // skip_frames, desc="Extraindo frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_id % skip_frames == 0:
                    filename = output_dir / f"frame_{frame_id:06d}.jpg"
                    cv2.imwrite(str(filename), frame)
                    saved_frames.append(str(filename))
                    pbar.update(1)
                
                frame_id += 1
            
            pbar.close()
            logger.info(f"{len(saved_frames)} frames extraídos")
            
            return saved_frames
            
        finally:
            cap.release()
