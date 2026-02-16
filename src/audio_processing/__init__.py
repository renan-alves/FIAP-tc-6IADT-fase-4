"""Pacote de processamento de áudio."""

from . import audio_analyzer
from .whisper_client import (
    HesitationMarker,
    TranscriptionResult,
    TranscriptionSegment,
    cleanup_temp_audio,
    save_audio_to_temp,
    transcribe_audio,
)

__all__ = [
    "audio_analyzer",
    "transcribe_audio",
    "TranscriptionResult",
    "TranscriptionSegment",
    "HesitationMarker",
    "save_audio_to_temp",
    "cleanup_temp_audio",
]
