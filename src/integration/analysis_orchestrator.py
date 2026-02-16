"""Orquestrador de análise DPP para detecção de Depressão Pós-Parto.

Este módulo integra transcrição (Whisper), extração de features acústicas
(Librosa - placeholder), e síntese clínica (GPT-4) para produzir uma
avaliação estruturada de risco de DPP.

Usado tanto pelo CLI (execução síncrona) quanto pela API (jobs assíncronos).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from config.logger import get_logger



logger = get_logger(__name__)


@dataclass
class StageTimings:
    """Registro de tempos de cada estágio do pipeline."""

    transcription_seconds: float = 0.0
    acoustic_seconds: float = 0.0
    synthesis_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "transcription_seconds": round(self.transcription_seconds, 2),
            "acoustic_seconds": round(self.acoustic_seconds, 2),
            "synthesis_seconds": round(self.synthesis_seconds, 2),
            "total_seconds": round(self.total_seconds, 2),
        }


def _generate_correlation_id() -> str:
    """Gera correlation ID único para rastreamento de request."""
    return str(uuid.uuid4())


def _extract_acoustic_data_from_transcription(transcription_result) -> Dict[str, Any]:
    """Extrai dados acústicos disponíveis do resultado de transcrição.

    Nota: Em futuras versões, isso será substituído por extração
    via Librosa (GOAL-002). Por agora, usamos dados da transcrição.
    """
    return {
        "hesitation_markers": [
            m.to_dict() for m in transcription_result.hesitation_markers
        ],
        "confidence": transcription_result.confidence,
        "duration": transcription_result.duration,
    }


def analyze_consultation(
    consultation_id: str | None,
    audio_path: Path,
    no_cleanup: bool = False,
    correlation_id: str | None = None,
) -> dict:
    """Executa análise completa de DPP para uma consulta de áudio.

    Pipeline:
    1. Transcrição via OpenAI Whisper
    2. Extração de features acústicas (placeholder - usa dados da transcrição)
    3. Síntese clínica via GPT-4

    Args:
        consultation_id: ID da consulta (gerado se não fornecido).
        audio_path: Caminho para o arquivo de áudio (.wav ou .mp3).
        no_cleanup: Se True, não remove arquivos temporários.
        correlation_id: ID de correlação para rastreamento (gerado se não fornecido).

    Returns:
        Dict com resultado da análise DPP seguindo o schema da spec.

    Raises:
        FileNotFoundError: Se o arquivo de áudio não existir.
        ValueError: Se o arquivo de áudio for inválido.
        RuntimeError: Se a análise falhar após retries.
    """
    from src.audio_processing.whisper_client import transcribe_audio
    from src.text_processing.gpt4_synthesizer import (
        create_fallback_assessment,
        synthesize_dpp_analysis,
    )
    from src.integration.audit_log import get_audit_logger

    start_time = time.time()
    timings = StageTimings()

    # Gerar IDs se não fornecidos
    correlation_id = correlation_id or _generate_correlation_id()
    consultation_id = consultation_id or f"consul-{uuid.uuid4().hex[:8]}"

    logger.info(
        "Iniciando análise DPP: consultation_id=%s correlation_id=%s audio=%s",
        consultation_id,
        correlation_id,
        audio_path,
    )

    # Validar arquivo de entrada
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.error("Arquivo de áudio não encontrado: %s", audio_path)
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")

    # Iniciar auditoria
    audit = get_audit_logger()
    audit.start_analysis(correlation_id, consultation_id)

    transcription_result = None
    acoustic_data = None
    assessment = None
    error_code = None
    error_message = None

    try:
        # Estágio 1: Transcrição via Whisper
        logger.info("[1/3] Transcrevendo áudio via Whisper...")
        audit.log_event(correlation_id, "transcription_started")

        stage_start = time.time()
        transcription_result = transcribe_audio(audio_path)
        timings.transcription_seconds = time.time() - stage_start

        audit.log_event(
            correlation_id,
            "transcription_completed",
            {
                "duration_seconds": timings.transcription_seconds,
                "text_length": len(transcription_result.text),
                "confidence": transcription_result.confidence,
                "hesitation_count": len(transcription_result.hesitation_markers),
            },
        )

        logger.info(
            "Transcrição concluída: %d caracteres, confiança=%.2f%%, hesitações=%d",
            len(transcription_result.text),
            transcription_result.confidence * 100,
            len(transcription_result.hesitation_markers),
        )

        # Estágio 2: Extração de features acústicas
        logger.info("[2/3] Extraindo features acústicas...")
        audit.log_event(correlation_id, "acoustic_extraction_started")

        stage_start = time.time()
        acoustic_data = _extract_acoustic_data_from_transcription(transcription_result)
        timings.acoustic_seconds = time.time() - stage_start

        audit.log_event(
            correlation_id,
            "acoustic_extraction_completed",
            {"duration_seconds": timings.acoustic_seconds},
        )

        logger.info("Features acústicas extraídas em %.2fs", timings.acoustic_seconds)

        # Estágio 3: Síntese clínica via GPT-4
        logger.info("[3/3] Sintetizando análise clínica via GPT-4...")
        audit.log_event(correlation_id, "synthesis_started")

        stage_start = time.time()
        assessment = synthesize_dpp_analysis(
            transcription=transcription_result.text,
            acoustic_data=acoustic_data,
            consultation_id=consultation_id,
        )
        timings.synthesis_seconds = time.time() - stage_start

        audit.log_event(
            correlation_id,
            "synthesis_completed",
            {
                "duration_seconds": timings.synthesis_seconds,
                "probability": assessment.analise_dpp.probabilidade_percentual,
                "risk_level": assessment.analise_dpp.nivel_risco,
            },
        )

        logger.info(
            "Síntese concluída em %.2fs: probabilidade=%d%%, risco=%s",
            timings.synthesis_seconds,
            assessment.analise_dpp.probabilidade_percentual,
            assessment.analise_dpp.nivel_risco,
        )

    except FileNotFoundError:
        error_code = "AUDIO_NOT_FOUND"
        error_message = f"Arquivo de áudio não encontrado: {audio_path}"
        raise

    except ValueError as e:
        error_code = "INVALID_INPUT"
        error_message = str(e)
        logger.error("Erro de validação: %s", error_message)
        raise

    except RuntimeError as e:
        error_code = "SYNTHESIS_FAILED"
        error_message = str(e)
        logger.error("Falha na síntese GPT-4: %s", error_message)

        # Graceful degradation: criar avaliação de fallback
        if transcription_result:
            logger.warning("Usando avaliação de fallback devido a falha na síntese")
            assessment = create_fallback_assessment(
                consultation_id=consultation_id,
                transcription=transcription_result.text,
                error_message=error_message,
            )
        else:
            # Transcrição falhou, não há como criar fallback
            raise RuntimeError(
                f"Análise falhou antes de completar transcrição: {error_message}"
            ) from e

    except Exception as e:
        error_code = "UNEXPECTED_ERROR"
        error_message = str(e)
        logger.exception("Erro inesperado na análise: %s", e)

        # Graceful degradation
        if transcription_result:
            assessment = create_fallback_assessment(
                consultation_id=consultation_id,
                transcription=transcription_result.text,
                error_message=error_message,
            )
        else:
            raise RuntimeError(
                f"Análise falhou antes da transcrição: {error_message}"
            ) from e

    finally:
        timings.total_seconds = time.time() - start_time

    # Finalizar auditoria
    if assessment:
        audit.complete_analysis(
            correlation_id=correlation_id,
            probability_percent=assessment.analise_dpp.probabilidade_percentual,
            risk_level=assessment.analise_dpp.nivel_risco,
            sugerir_alerta=assessment.analise_dpp.sugerir_alerta,
            confidence=assessment.analise_dpp.confianca_analise,
            duration_seconds=timings.total_seconds,
        )
    else:
        audit.fail_analysis(
            correlation_id=correlation_id,
            error_code=error_code or "UNKNOWN",
            error_message=error_message or "Erro desconhecido",
            duration_seconds=timings.total_seconds,
        )

    # Construir resultado final
    result = assessment.to_dict() if assessment else {}

    # Adicionar metadados estendidos
    result["metadata"] = result.get("metadata", {})
    result["metadata"]["job_id"] = correlation_id
    result["metadata"]["correlation_id"] = correlation_id
    result["metadata"]["audio_path"] = str(audio_path)
    result["metadata"]["timings"] = timings.to_dict()

    logger.info(
        "Análise concluída: consultation_id=%s total=%.2fs risco=%s",
        consultation_id,
        timings.total_seconds,
        assessment.analise_dpp.nivel_risco if assessment else "N/A",
    )

    return result


def analyze_consultation_async(
    job_id: str,
    consultation_id: str,
    audio_path: Path,
    correlation_id: str,
) -> None:
    """Wrapper para execução assíncrona via job cache (usado pela API).

    Atualiza o job cache com o resultado ou erro da análise.

    Args:
        job_id: ID do job no cache.
        consultation_id: ID da consulta.
        audio_path: Caminho para o arquivo de áudio.
        correlation_id: ID de correlação para rastreamento.
    """
    from src.integration.job_cache import get_job_cache, JobStatus

    cache = get_job_cache()
    cache.update_status(job_id, JobStatus.PROCESSING)

    try:
        result = analyze_consultation(
            consultation_id=consultation_id,
            audio_path=audio_path,
            correlation_id=correlation_id,
        )
        cache.complete_job(job_id, result)

    except FileNotFoundError as e:
        cache.fail_job(job_id, "AUDIO_NOT_FOUND", str(e))

    except ValueError as e:
        cache.fail_job(job_id, "INVALID_INPUT", str(e))

    except RuntimeError as e:
        cache.fail_job(job_id, "ANALYSIS_FAILED", str(e))

    except Exception as e:
        logger.exception("Erro inesperado no job %s: %s", job_id, e)
        cache.fail_job(job_id, "UNEXPECTED_ERROR", str(e))
