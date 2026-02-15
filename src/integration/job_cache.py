"""Cache de jobs em memória para análises DPP.

Implementa armazenamento temporário de resultados de análise
para suporte a polling da API assíncrona.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Dict, Optional

from config.logger import get_logger

logger = get_logger(__name__)

# TTL padrão de 24 horas para jobs
DEFAULT_TTL_SECONDS = 24 * 60 * 60


class JobStatus(str, Enum):
    """Status possíveis de um job de análise."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalysisJob:
    """Representa um job de análise DPP."""

    job_id: str
    correlation_id: str
    consultation_id: str
    status: JobStatus
    created_at: str  # ISO-8601
    updated_at: str  # ISO-8601
    result: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    audio_path: Optional[str] = None
    expires_at: float = field(default_factory=lambda: time.time() + DEFAULT_TTL_SECONDS)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "correlation_id": self.correlation_id,
            "consultation_id": self.consultation_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


class JobCache:
    """Cache em memória para jobs de análise.

    Thread-safe e com suporte a TTL para expiração automática.
    Usado pela API para armazenar resultados de jobs assíncronos.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._cache: Dict[str, AnalysisJob] = {}
        self._lock = Lock()
        self._ttl_seconds = ttl_seconds
        logger.info("JobCache inicializado com TTL=%ds", ttl_seconds)

    def _get_timestamp(self) -> str:
        """Retorna timestamp ISO-8601 UTC."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _cleanup_expired(self) -> int:
        """Remove jobs expirados do cache. Retorna quantidade removida."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if v.expires_at < now]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.info("JobCache: removidos %d jobs expirados", len(expired_keys))
        return len(expired_keys)

    def create_job(
        self,
        job_id: str,
        correlation_id: str,
        consultation_id: str,
        audio_path: Optional[str] = None,
    ) -> AnalysisJob:
        """Cria um novo job no cache."""
        timestamp = self._get_timestamp()

        job = AnalysisJob(
            job_id=job_id,
            correlation_id=correlation_id,
            consultation_id=consultation_id,
            status=JobStatus.PENDING,
            created_at=timestamp,
            updated_at=timestamp,
            audio_path=audio_path,
            expires_at=time.time() + self._ttl_seconds,
        )

        with self._lock:
            self._cleanup_expired()
            self._cache[job_id] = job

        logger.info("JobCache: criado job_id=%s", job_id)
        return job

    def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        """Recupera job pelo ID."""
        with self._lock:
            self._cleanup_expired()
            job = self._cache.get(job_id)

        if job and job.expires_at < time.time():
            return None

        return job

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
    ) -> Optional[AnalysisJob]:
        """Atualiza o status de um job."""
        with self._lock:
            job = self._cache.get(job_id)
            if not job:
                logger.warning("JobCache: job_id=%s não encontrado para update", job_id)
                return None

            job.status = status
            job.updated_at = self._get_timestamp()
            logger.debug("JobCache: job_id=%s status=%s", job_id, status.value)

        return job

    def complete_job(
        self,
        job_id: str,
        result: Dict[str, Any],
    ) -> Optional[AnalysisJob]:
        """Marca job como completo com resultado."""
        with self._lock:
            job = self._cache.get(job_id)
            if not job:
                logger.warning(
                    "JobCache: job_id=%s não encontrado para complete", job_id
                )
                return None

            job.status = JobStatus.COMPLETED
            job.result = result
            job.updated_at = self._get_timestamp()
            logger.info("JobCache: job_id=%s completo", job_id)

        return job

    def fail_job(
        self,
        job_id: str,
        error_code: str,
        error_message: str,
    ) -> Optional[AnalysisJob]:
        """Marca job como falho com erro."""
        with self._lock:
            job = self._cache.get(job_id)
            if not job:
                logger.warning("JobCache: job_id=%s não encontrado para fail", job_id)
                return None

            job.status = JobStatus.FAILED
            job.error_code = error_code
            job.error_message = error_message
            job.updated_at = self._get_timestamp()
            logger.info("JobCache: job_id=%s falhou com %s", job_id, error_code)

        return job

    def delete_job(self, job_id: str) -> bool:
        """Remove job do cache."""
        with self._lock:
            if job_id in self._cache:
                del self._cache[job_id]
                logger.debug("JobCache: job_id=%s removido", job_id)
                return True
        return False

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        consultation_id: Optional[str] = None,
    ) -> list:
        """Lista jobs com filtros opcionais."""
        with self._lock:
            self._cleanup_expired()
            jobs = list(self._cache.values())

        if status:
            jobs = [j for j in jobs if j.status == status]
        if consultation_id:
            jobs = [j for j in jobs if j.consultation_id == consultation_id]

        return jobs

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        with self._lock:
            self._cleanup_expired()
            total = len(self._cache)
            by_status = {}
            for job in self._cache.values():
                status = job.status.value
                by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_jobs": total,
            "by_status": by_status,
            "ttl_seconds": self._ttl_seconds,
        }


# Instância global do job cache
_job_cache: Optional[JobCache] = None


def get_job_cache() -> JobCache:
    """Retorna a instância global do job cache."""
    global _job_cache
    if _job_cache is None:
        _job_cache = JobCache()
    return _job_cache
