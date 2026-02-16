"""Cliente OpenAI Whisper para transcrição de áudio em português.

Este módulo integra a API OpenAI Whisper para transcrição de consultas clínicas,
com detecção de marcadores de hesitação (pausas e filler words) relevantes para
análise de Depressão Pós-Parto (DPP).
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from config.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Filler words comuns em português que indicam hesitação
FILLER_WORDS_PT = {
    "uh",
    "uhm",
    "um",
    "hm",
    "hmm",
    "é",
    "éh",
    "eh",
    "ah",
    "ahm",
    "então",
    "tipo",
    "né",
    "sabe",
    "assim",
    "bom",
    "bem",
    "olha",
    "veja",
    "quer dizer",
    "na verdade",
    "digamos",
}

# Pause threshold em segundos para marcador de hesitação
PAUSE_THRESHOLD_SECONDS = 1.5

# Configuração de retry
MAX_RETRIES = 3
BACKOFF_BASE_MS = 100  # 100ms, 200ms, 400ms


@dataclass
class HesitationMarker:
    """Marcador de hesitação detectado na transcrição."""

    type: str  # "pause" ou "filler_word"
    start_time: float  # Timestamp de início em segundos
    end_time: float  # Timestamp de fim em segundos
    text: Optional[str] = None  # Texto do filler word (se aplicável)
    duration: float = 0.0  # Duração em segundos (para pausas)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "duration": self.duration,
        }


@dataclass
class TranscriptionSegment:
    """Segmento de transcrição com timestamps."""

    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass
class TranscriptionResult:
    """Resultado da transcrição de áudio com marcadores de hesitação.

    Attributes:
        text: Texto completo da transcrição.
        segments: Lista de segmentos com timestamps.
        hesitation_markers: Marcadores de hesitação detectados.
        confidence: Confiança média da transcrição (0.0-1.0).
        language: Idioma detectado.
        duration: Duração total do áudio em segundos.
    """

    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    hesitation_markers: List[HesitationMarker] = field(default_factory=list)
    confidence: float = 0.0
    language: str = "pt"
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "hesitation_markers": [h.to_dict() for h in self.hesitation_markers],
            "confidence": self.confidence,
            "language": self.language,
            "duration": self.duration,
        }


def _get_openai_client():
    """Obtém cliente OpenAI configurado com API key do ambiente."""
    try:
        from openai import OpenAI
    except ImportError as e:
        logger.error("OpenAI SDK não instalado: %s", e)
        raise ImportError(
            "OpenAI SDK requerido. Instale com: pip install openai>=1.0"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY não configurada no ambiente")
        raise ValueError("OPENAI_API_KEY environment variable is required")

    return OpenAI(api_key=api_key)


def _detect_filler_words(
    text: str, segment_start: float, segment_end: float
) -> List[HesitationMarker]:
    """Detecta filler words no texto do segmento."""
    markers = []
    words = text.lower().split()
    segment_duration = segment_end - segment_start
    word_duration = segment_duration / max(len(words), 1)

    for i, word in enumerate(words):
        clean_word = re.sub(r"[^\w]", "", word)
        if clean_word in FILLER_WORDS_PT:
            word_start = segment_start + (i * word_duration)
            word_end = word_start + word_duration
            markers.append(
                HesitationMarker(
                    type="filler_word",
                    start_time=word_start,
                    end_time=word_end,
                    text=clean_word,
                    duration=word_duration,
                )
            )
    return markers


def _detect_pauses(segments: List[TranscriptionSegment]) -> List[HesitationMarker]:
    """Detecta pausas longas entre segmentos (>1.5s)."""
    markers = []
    for i in range(1, len(segments)):
        prev_end = segments[i - 1].end
        curr_start = segments[i].start
        gap = curr_start - prev_end

        if gap >= PAUSE_THRESHOLD_SECONDS:
            markers.append(
                HesitationMarker(
                    type="pause",
                    start_time=prev_end,
                    end_time=curr_start,
                    text=None,
                    duration=gap,
                )
            )
    return markers


def _calculate_confidence(response) -> float:
    """Calcula confiança média baseada em avg_logprob dos segmentos."""
    if not hasattr(response, "segments") or not response.segments:
        return 0.8  # Default quando não há segmentos detalhados

    total_logprob = 0.0
    count = 0
    for seg in response.segments:
        if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
            total_logprob += seg.avg_logprob
            count += 1

    if count == 0:
        return 0.8

    avg_logprob = total_logprob / count
    # Converter log-prob para escala 0-1 (logprob típico varia de -1 a 0)
    confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
    return round(confidence, 3)


def transcribe_audio(
    audio_path: Path,
    language: str = "pt",
    prompt: Optional[str] = None,
) -> TranscriptionResult:
    """Transcreve áudio usando OpenAI Whisper com detecção de hesitação.

    Args:
        audio_path: Caminho para o arquivo de áudio (.wav, .mp3, .mp4, etc.).
        language: Código do idioma (default: "pt" para português).
        prompt: Prompt opcional para melhorar a transcrição.

    Returns:
        TranscriptionResult com texto, segmentos e marcadores de hesitação.

    Raises:
        FileNotFoundError: Se o arquivo de áudio não existir.
        ValueError: Se OPENAI_API_KEY não estiver configurada.
        RuntimeError: Se a transcrição falhar após todas as tentativas.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    client = _get_openai_client()
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                "Iniciando transcrição (tentativa %d/%d): %s",
                attempt + 1,
                MAX_RETRIES,
                audio_path.name,
            )

            with open(audio_path, "rb") as audio_file:
                # Usar whisper-1 com verbose_json para obter timestamps de segmentos
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    prompt=prompt
                    or "Transcrição de consulta clínica em português brasileiro.",
                )

            # Extrair segmentos
            segments = []
            if hasattr(response, "segments") and response.segments:
                for i, seg in enumerate(response.segments):
                    segments.append(
                        TranscriptionSegment(
                            id=i,
                            start=seg.start,
                            end=seg.end,
                            text=seg.text.strip(),
                        )
                    )

            # Detectar marcadores de hesitação
            hesitation_markers = []

            # Detectar pausas entre segmentos
            hesitation_markers.extend(_detect_pauses(segments))

            # Detectar filler words em cada segmento
            for seg in segments:
                hesitation_markers.extend(
                    _detect_filler_words(seg.text, seg.start, seg.end)
                )

            # Ordenar marcadores por timestamp
            hesitation_markers.sort(key=lambda m: m.start_time)

            # Calcular confiança e duração
            confidence = _calculate_confidence(response)
            duration = getattr(response, "duration", 0.0)
            if not duration and segments:
                duration = segments[-1].end

            result = TranscriptionResult(
                text=response.text,
                segments=segments,
                hesitation_markers=hesitation_markers,
                confidence=confidence,
                language=language,
                duration=duration,
            )

            logger.info(
                "Transcrição concluída: %d caracteres, %d segmentos, %d marcadores de hesitação",
                len(result.text),
                len(result.segments),
                len(result.hesitation_markers),
            )

            return result

        except Exception as e:
            last_error = e
            logger.warning(
                "Erro na transcrição (tentativa %d/%d): %s",
                attempt + 1,
                MAX_RETRIES,
                str(e),
            )

            if attempt < MAX_RETRIES - 1:
                # Exponential backoff: 100ms, 200ms, 400ms
                wait_ms = BACKOFF_BASE_MS * (2**attempt)
                logger.info("Aguardando %dms antes de retry...", wait_ms)
                time.sleep(wait_ms / 1000.0)

    # Todas as tentativas falharam
    error_msg = f"Transcrição falhou após {MAX_RETRIES} tentativas: {last_error}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_error


def save_audio_to_temp(audio_bytes: bytes, suffix: str = ".wav") -> Path:
    """Salva bytes de áudio em arquivo temporário.

    Args:
        audio_bytes: Conteúdo do arquivo de áudio.
        suffix: Extensão do arquivo (default: .wav).

    Returns:
        Path para o arquivo temporário criado.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(tmp_path)
    path.write_bytes(audio_bytes)
    logger.debug("Áudio salvo em arquivo temporário: %s", path)
    return path


def cleanup_temp_audio(path: Path) -> None:
    """Remove arquivo de áudio temporário.

    Args:
        path: Caminho do arquivo a ser removido.
    """
    try:
        if path.exists():
            path.unlink()
            logger.debug("Arquivo temporário removido: %s", path)
    except OSError as e:
        logger.warning("Erro ao remover arquivo temporário %s: %s", path, e)
