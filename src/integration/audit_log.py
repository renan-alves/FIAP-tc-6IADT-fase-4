"""Módulo de logging de auditoria para análises DPP.

Implementa structured logging em JSON para compliance LGPD,
mantendo registro de análises sem dados sensíveis (PII).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config.logger import get_logger
from config.settings import DATA_DIR

logger = get_logger(__name__)

AUDIT_LOG_DIR = DATA_DIR / "audit_logs"
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AuditEvent:
    """Evento de auditoria para análise DPP."""

    correlation_id: str
    event_type: str  # started, transcribed, synthesized, completed, failed
    timestamp: str  # ISO-8601
    consultation_id: str
    details: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditLogEntry:
    """Entrada completa de log de auditoria para uma análise."""

    correlation_id: str
    consultation_id: str
    started_at: str
    completed_at: Optional[str]
    status: str  # processing, completed, failed
    probability_percent: Optional[int]
    risk_level: Optional[str]
    sugerir_alerta: Optional[bool]
    confidence: Optional[float]
    duration_seconds: Optional[float]
    events: list

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLogger:
    """Logger de auditoria para análises DPP.

    Mantém logs estruturados em JSON sem PII para compliance LGPD.
    """

    def __init__(self, log_dir: Path = AUDIT_LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_entries: Dict[str, AuditLogEntry] = {}
        logger.info("AuditLogger inicializado em %s", self.log_dir)

    def _get_timestamp(self) -> str:
        """Retorna timestamp ISO-8601 UTC."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _get_log_file_path(self) -> Path:
        """Retorna path do arquivo de log do dia atual."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def _append_to_log_file(self, entry: Dict[str, Any]) -> None:
        """Adiciona entrada ao arquivo de log (JSON Lines format)."""
        log_path = self._get_log_file_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def start_analysis(
        self,
        correlation_id: str,
        consultation_id: str,
    ) -> None:
        """Registra início de uma análise."""
        timestamp = self._get_timestamp()

        entry = AuditLogEntry(
            correlation_id=correlation_id,
            consultation_id=consultation_id,
            started_at=timestamp,
            completed_at=None,
            status="processing",
            probability_percent=None,
            risk_level=None,
            sugerir_alerta=None,
            confidence=None,
            duration_seconds=None,
            events=[],
        )

        event = AuditEvent(
            correlation_id=correlation_id,
            event_type="started",
            timestamp=timestamp,
            consultation_id=consultation_id,
            details={"action": "analysis_started"},
        )

        entry.events.append(event.to_dict())
        self._current_entries[correlation_id] = entry

        logger.info(
            "Auditoria: início análise correlation_id=%s consultation_id=%s",
            correlation_id,
            consultation_id,
        )

    def log_event(
        self,
        correlation_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra evento intermediário na análise."""
        if correlation_id not in self._current_entries:
            logger.warning(
                "Tentativa de log para correlation_id desconhecido: %s", correlation_id
            )
            return

        entry = self._current_entries[correlation_id]
        event = AuditEvent(
            correlation_id=correlation_id,
            event_type=event_type,
            timestamp=self._get_timestamp(),
            consultation_id=entry.consultation_id,
            details=details or {},
        )

        entry.events.append(event.to_dict())

        logger.debug(
            "Auditoria: evento %s para correlation_id=%s",
            event_type,
            correlation_id,
        )

    def complete_analysis(
        self,
        correlation_id: str,
        probability_percent: int,
        risk_level: str,
        sugerir_alerta: bool,
        confidence: float,
        duration_seconds: float,
    ) -> None:
        """Registra conclusão bem-sucedida de uma análise."""
        if correlation_id not in self._current_entries:
            logger.warning(
                "Tentativa de completar correlation_id desconhecido: %s", correlation_id
            )
            return

        entry = self._current_entries[correlation_id]
        timestamp = self._get_timestamp()

        entry.completed_at = timestamp
        entry.status = "completed"
        entry.probability_percent = probability_percent
        entry.risk_level = risk_level
        entry.sugerir_alerta = sugerir_alerta
        entry.confidence = confidence
        entry.duration_seconds = duration_seconds

        event = AuditEvent(
            correlation_id=correlation_id,
            event_type="completed",
            timestamp=timestamp,
            consultation_id=entry.consultation_id,
            details={
                "probability_percent": probability_percent,
                "risk_level": risk_level,
                "sugerir_alerta": sugerir_alerta,
                "duration_seconds": duration_seconds,
            },
        )

        entry.events.append(event.to_dict())

        # Persistir entrada no arquivo de log
        self._append_to_log_file(entry.to_dict())

        # Remover da memória
        del self._current_entries[correlation_id]

        logger.info(
            "Auditoria: análise concluída correlation_id=%s risco=%s probabilidade=%d%%",
            correlation_id,
            risk_level,
            probability_percent,
        )

    def fail_analysis(
        self,
        correlation_id: str,
        error_code: str,
        error_message: str,
        duration_seconds: float,
    ) -> None:
        """Registra falha de uma análise."""
        if correlation_id not in self._current_entries:
            logger.warning(
                "Tentativa de falhar correlation_id desconhecido: %s", correlation_id
            )
            return

        entry = self._current_entries[correlation_id]
        timestamp = self._get_timestamp()

        entry.completed_at = timestamp
        entry.status = "failed"
        entry.duration_seconds = duration_seconds

        event = AuditEvent(
            correlation_id=correlation_id,
            event_type="failed",
            timestamp=timestamp,
            consultation_id=entry.consultation_id,
            details={
                "error_code": error_code,
                "error_message": error_message,
                "duration_seconds": duration_seconds,
            },
        )

        entry.events.append(event.to_dict())

        # Persistir entrada no arquivo de log
        self._append_to_log_file(entry.to_dict())

        # Remover da memória
        del self._current_entries[correlation_id]

        logger.info(
            "Auditoria: análise falhou correlation_id=%s erro=%s",
            correlation_id,
            error_code,
        )


# Instância global do audit logger
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Retorna a instância global do audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
