"""Pydantic models for DPP Audio Analysis API.

Define request/response schemas per spec v1.2.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Enums ---


class RiskLevel(str, Enum):
    """Níveis de risco para DPP."""

    BAIXO = "Baixo"
    MODERADO = "Moderado"
    ALTO = "Alto"
    CRITICO = "Crítico"


class JobStatusEnum(str, Enum):
    """Status possíveis de um job de análise."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AudioFormat(str, Enum):
    """Formatos de áudio suportados pela API."""

    WAV = "wav"
    MP3 = "mp3"
    MP4 = "mp4"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"
    WEBM = "webm"
    MPEG = "mpeg"
    MPGA = "mpga"
    OGA = "oga"


# --- Request Models ---


class AnalyzeRequest(BaseModel):
    """Request model para POST /analyze."""

    consultation_id: Optional[str] = Field(
        default=None,
        description="Identificador único da consulta",
        examples=["consul-2026-02-14-001"],
    )
    audio_format: Optional[AudioFormat] = Field(
        default=None,
        description="Formato do arquivo de áudio (detectado automaticamente se omitido)",
    )


# --- Response Models ---


class AnalyzeAccepted(BaseModel):
    """Response para POST /analyze (HTTP 202)."""

    job_id: str = Field(description="UUID do job criado")
    correlation_id: str = Field(description="UUID para rastreamento de requisição")


class AnalyzeStatusProcessing(BaseModel):
    """Response para GET /analyze/{job_id} quando job ainda está processando."""

    status: str = Field(default="processing", description="Status atual do job")


class ErrorEnvelope(BaseModel):
    """Envelope padrão para erros da API."""

    error_code: str = Field(
        description="Código do erro para processamento programático"
    )
    message: str = Field(description="Mensagem legível para humanos")
    retry_after: Optional[int] = Field(
        default=None, description="Segundos para retry (quando aplicável)"
    )


# --- DPP Risk Assessment Models ---


class ComponenteAnalise(BaseModel):
    """Pesos dos componentes da análise."""

    componente_textual_peso: float = Field(
        ge=0, le=1, description="Peso do componente textual (0-1)"
    )
    componente_acustico_peso: float = Field(
        ge=0, le=1, description="Peso do componente acústico (0-1)"
    )


class AnaliseDpp(BaseModel):
    """Resultado estruturado da análise de DPP."""

    probabilidade_percentual: int = Field(
        ge=0, le=100, description="Probabilidade de DPP (0-100%)"
    )
    nivel_risco: RiskLevel = Field(description="Classificação de risco")
    indicadores_detectados: List[str] = Field(
        default_factory=list, description="Lista de indicadores clínicos detectados"
    )
    sugerir_alerta: bool = Field(
        description="Flag para sugestão de alerta (true se >= 75%)"
    )
    justificativa_clinica: str = Field(description="Justificativa clínica em português")
    confianca_analise: float = Field(
        ge=0, le=1, description="Confiança da análise (0.0-1.0)"
    )
    componentes_analise: ComponenteAnalise = Field(description="Pesos dos componentes")


class AnalysisMetadata(BaseModel):
    """Metadados da análise."""

    transcription_confidence: float = Field(
        ge=0, le=1, description="Confiança da transcrição"
    )
    acoustic_features_available: bool = Field(
        description="Indica se features acústicas foram extraídas"
    )
    analysis_duration_seconds: float = Field(
        ge=0, description="Duração total da análise em segundos"
    )
    quality_warning: Optional[str] = Field(
        default=None, description="Aviso de qualidade (se aplicável)"
    )


class DppRiskAssessment(BaseModel):
    """Response completo para análise de DPP concluída."""

    consultation_id: Optional[str] = Field(default=None, description="ID da consulta")
    analysis_timestamp: str = Field(description="Timestamp ISO-8601 da análise")
    analise_dpp: AnaliseDpp = Field(description="Resultado da análise de DPP")
    metadata: AnalysisMetadata = Field(description="Metadados da análise")


class DppRiskAssessmentResponse(BaseModel):
    """Response wrapper para GET /analyze/{job_id} quando completo."""

    consultation_id: Optional[str] = None
    analysis_timestamp: str
    analise_dpp: Dict[str, Any]
    metadata: Dict[str, Any]


# --- Health Check Models ---


class DependencyStatus(BaseModel):
    """Status de uma dependência."""

    name: str
    status: str = Field(description="ok | degraded | unavailable")
    latency_ms: Optional[float] = Field(default=None, description="Latência em ms")
    message: Optional[str] = Field(
        default=None, description="Mensagem de erro se houver"
    )


class HealthResponse(BaseModel):
    """Response para GET /health."""

    status: str = Field(description="ok | degraded | unhealthy")
    timestamp: str = Field(description="Timestamp ISO-8601")
    version: str = Field(default="1.0.0", description="Versão da API")
    dependencies: List[DependencyStatus] = Field(
        default_factory=list, description="Status das dependências"
    )


# --- Job Status Models ---


class JobStatusResponse(BaseModel):
    """Response para GET /analyze/{job_id} com status do job."""

    job_id: str
    status: JobStatusEnum
    correlation_id: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
