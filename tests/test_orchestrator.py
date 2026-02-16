"""Testes para o orquestrador de análise DPP e módulos de suporte."""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAuditLog:
    """Testes para o módulo de audit logging."""

    def test_audit_logger_initialization(self):
        """AuditLogger deve inicializar com diretório de logs."""
        from src.integration.audit_log import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "audit"
            logger = AuditLogger(log_dir)
            assert log_dir.exists()

    def test_start_analysis_creates_entry(self):
        """start_analysis deve criar entrada no logger."""
        from src.integration.audit_log import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir))
            logger.start_analysis("corr-123", "consul-456")

            assert "corr-123" in logger._current_entries
            entry = logger._current_entries["corr-123"]
            assert entry.consultation_id == "consul-456"
            assert entry.status == "processing"

    def test_log_event_appends_to_entry(self):
        """log_event deve adicionar evento à entrada."""
        from src.integration.audit_log import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir))
            logger.start_analysis("corr-123", "consul-456")
            logger.log_event("corr-123", "transcription_started", {"test": "data"})

            entry = logger._current_entries["corr-123"]
            assert len(entry.events) == 2
            assert entry.events[1]["event_type"] == "transcription_started"

    def test_complete_analysis_persists_to_file(self):
        """complete_analysis deve persistir entrada no arquivo de log."""
        from src.integration.audit_log import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir))
            logger.start_analysis("corr-123", "consul-456")
            logger.complete_analysis(
                correlation_id="corr-123",
                probability_percent=65,
                risk_level="Alto",
                sugerir_alerta=False,
                confidence=0.85,
                duration_seconds=10.5,
            )

            # Entrada deve ser removida da memória
            assert "corr-123" not in logger._current_entries

            # Arquivo deve existir e conter o log
            log_files = list(Path(tmpdir).glob("audit_*.jsonl"))
            assert len(log_files) == 1

            with open(log_files[0], "r", encoding="utf-8") as f:
                line = f.readline()
                data = json.loads(line)
                assert data["correlation_id"] == "corr-123"
                assert data["probability_percent"] == 65
                assert data["risk_level"] == "Alto"

    def test_fail_analysis_persists_error(self):
        """fail_analysis deve persistir erro no arquivo de log."""
        from src.integration.audit_log import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir))
            logger.start_analysis("corr-123", "consul-456")
            logger.fail_analysis(
                correlation_id="corr-123",
                error_code="TRANSCRIPTION_FAILED",
                error_message="API timeout",
                duration_seconds=5.0,
            )

            log_files = list(Path(tmpdir).glob("audit_*.jsonl"))
            with open(log_files[0], "r", encoding="utf-8") as f:
                data = json.loads(f.readline())
                assert data["status"] == "failed"
                assert "TRANSCRIPTION_FAILED" in str(data)


class TestJobCache:
    """Testes para o módulo de job cache."""

    def test_create_job(self):
        """create_job deve criar job com status pending."""
        from src.integration.job_cache import JobCache, JobStatus

        cache = JobCache(ttl_seconds=3600)
        job = cache.create_job("job-1", "corr-1", "consul-1")

        assert job.job_id == "job-1"
        assert job.status == JobStatus.PENDING
        assert job.consultation_id == "consul-1"

    def test_get_job_returns_created_job(self):
        """get_job deve retornar job criado anteriormente."""
        from src.integration.job_cache import JobCache

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")
        job = cache.get_job("job-1")

        assert job is not None
        assert job.job_id == "job-1"

    def test_get_job_returns_none_for_unknown(self):
        """get_job deve retornar None para job desconhecido."""
        from src.integration.job_cache import JobCache

        cache = JobCache()
        job = cache.get_job("nonexistent")

        assert job is None

    def test_update_status(self):
        """update_status deve atualizar status do job."""
        from src.integration.job_cache import JobCache, JobStatus

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")
        cache.update_status("job-1", JobStatus.PROCESSING)

        job = cache.get_job("job-1")
        assert job.status == JobStatus.PROCESSING

    def test_complete_job(self):
        """complete_job deve marcar job como completo com resultado."""
        from src.integration.job_cache import JobCache, JobStatus

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")

        result = {"analise_dpp": {"probabilidade_percentual": 50}}
        cache.complete_job("job-1", result)

        job = cache.get_job("job-1")
        assert job.status == JobStatus.COMPLETED
        assert job.result == result

    def test_fail_job(self):
        """fail_job deve marcar job como falho com erro."""
        from src.integration.job_cache import JobCache, JobStatus

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")
        cache.fail_job("job-1", "API_ERROR", "Timeout")

        job = cache.get_job("job-1")
        assert job.status == JobStatus.FAILED
        assert job.error_code == "API_ERROR"
        assert job.error_message == "Timeout"

    def test_delete_job(self):
        """delete_job deve remover job do cache."""
        from src.integration.job_cache import JobCache

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")
        result = cache.delete_job("job-1")

        assert result is True
        assert cache.get_job("job-1") is None

    def test_list_jobs_filters_by_status(self):
        """list_jobs deve filtrar por status."""
        from src.integration.job_cache import JobCache, JobStatus

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")
        cache.create_job("job-2", "corr-2", "consul-2")
        cache.update_status("job-2", JobStatus.PROCESSING)

        pending_jobs = cache.list_jobs(status=JobStatus.PENDING)
        processing_jobs = cache.list_jobs(status=JobStatus.PROCESSING)

        assert len(pending_jobs) == 1
        assert len(processing_jobs) == 1

    def test_get_stats(self):
        """get_stats deve retornar estatísticas do cache."""
        from src.integration.job_cache import JobCache, JobStatus

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")
        cache.create_job("job-2", "corr-2", "consul-2")
        cache.update_status("job-2", JobStatus.COMPLETED)

        stats = cache.get_stats()

        assert stats["total_jobs"] == 2
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["completed"] == 1

    def test_expired_jobs_are_cleaned_up(self):
        """Jobs expirados devem ser removidos automaticamente."""
        from src.integration.job_cache import JobCache

        cache = JobCache(ttl_seconds=1)
        cache.create_job("job-1", "corr-1", "consul-1")

        time.sleep(1.1)

        job = cache.get_job("job-1")
        assert job is None


class TestOrchestratorUnit:
    """Testes unitários para o orquestrador (com mocks)."""

    def test_analyze_consultation_file_not_found(self):
        """analyze_consultation deve levantar erro se arquivo não existe."""
        from src.integration.analysis_orchestrator import analyze_consultation

        with pytest.raises(FileNotFoundError):
            analyze_consultation("consul-1", Path("/nonexistent/file.wav"))

    @patch("src.integration.audit_log.get_audit_logger")
    @patch("src.audio_processing.whisper_client.transcribe_audio")
    @patch("src.text_processing.gpt4_synthesizer.synthesize_dpp_analysis")
    def test_analyze_consultation_success_flow(
        self, mock_synthesize, mock_transcribe, mock_audit
    ):
        """analyze_consultation deve executar pipeline completo."""
        # Reimport to get fresh module
        import importlib
        import src.integration.analysis_orchestrator as orch_module

        importlib.reload(orch_module)

        from src.text_processing.gpt4_synthesizer import (
            AnaliseDPP,
            ComponenteAnalise,
            DPPAssessmentMetadata,
            DPPRiskAssessment,
        )
        from src.audio_processing.whisper_client import TranscriptionResult

        # Setup mocks
        mock_audit_instance = MagicMock()
        mock_audit.return_value = mock_audit_instance

        mock_transcription = TranscriptionResult(
            text="Texto de teste",
            segments=[],
            hesitation_markers=[],
            confidence=0.9,
            language="pt",
            duration=30.0,
        )
        mock_transcribe.return_value = mock_transcription

        mock_assessment = DPPRiskAssessment(
            consultation_id="consul-1",
            analysis_timestamp="2026-02-14T10:00:00Z",
            analise_dpp=AnaliseDPP(
                probabilidade_percentual=45,
                nivel_risco="Moderado",
                indicadores_detectados=["fadiga"],
                sugerir_alerta=False,
                justificativa_clinica="Teste",
                confianca_analise=0.85,
                componentes_analise=ComponenteAnalise(
                    componente_textual_peso=0.7,
                    componente_acustico_peso=0.3,
                ),
            ),
            metadata=DPPAssessmentMetadata(
                transcription_confidence=0.9,
                acoustic_features_available=True,
                analysis_duration_seconds=5.0,
            ),
        )
        mock_synthesize.return_value = mock_assessment

        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = orch_module.analyze_consultation("consul-1", temp_path)

            # Verify pipeline executed
            mock_transcribe.assert_called_once()
            mock_synthesize.assert_called_once()
            mock_audit_instance.start_analysis.assert_called_once()
            mock_audit_instance.complete_analysis.assert_called_once()

            # Verify result structure
            assert "analise_dpp" in result
            assert result["analise_dpp"]["probabilidade_percentual"] == 45
            assert result["metadata"]["timings"]["total_seconds"] >= 0

        finally:
            temp_path.unlink()

    @patch("src.integration.audit_log.get_audit_logger")
    @patch("src.audio_processing.whisper_client.transcribe_audio")
    @patch("src.text_processing.gpt4_synthesizer.synthesize_dpp_analysis")
    @patch("src.text_processing.gpt4_synthesizer.create_fallback_assessment")
    def test_analyze_consultation_fallback_on_synthesis_error(
        self, mock_fallback, mock_synthesize, mock_transcribe, mock_audit
    ):
        """analyze_consultation deve usar fallback se síntese falhar."""
        import importlib
        import src.integration.analysis_orchestrator as orch_module

        importlib.reload(orch_module)

        from src.text_processing.gpt4_synthesizer import (
            AnaliseDPP,
            ComponenteAnalise,
            DPPAssessmentMetadata,
            DPPRiskAssessment,
        )
        from src.audio_processing.whisper_client import TranscriptionResult

        mock_audit_instance = MagicMock()
        mock_audit.return_value = mock_audit_instance

        mock_transcription = TranscriptionResult(
            text="Texto de teste",
            segments=[],
            hesitation_markers=[],
            confidence=0.9,
            language="pt",
            duration=30.0,
        )
        mock_transcribe.return_value = mock_transcription

        mock_synthesize.side_effect = RuntimeError("API Error")

        mock_fallback_assessment = DPPRiskAssessment(
            consultation_id="consul-1",
            analysis_timestamp="2026-02-14T10:00:00Z",
            analise_dpp=AnaliseDPP(
                probabilidade_percentual=0,
                nivel_risco="Baixo",
                indicadores_detectados=[],
                sugerir_alerta=False,
                justificativa_clinica="Fallback",
                confianca_analise=0.0,
                componentes_analise=ComponenteAnalise(
                    componente_textual_peso=0.0,
                    componente_acustico_peso=0.0,
                ),
            ),
            metadata=DPPAssessmentMetadata(
                transcription_confidence=0.0,
                acoustic_features_available=False,
                analysis_duration_seconds=0.0,
                quality_warning="Fallback: API Error",
            ),
        )
        mock_fallback.return_value = mock_fallback_assessment

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = orch_module.analyze_consultation("consul-1", temp_path)

            mock_fallback.assert_called_once()
            assert result["analise_dpp"]["probabilidade_percentual"] == 0
            assert "Fallback" in str(result["metadata"].get("quality_warning", ""))

        finally:
            temp_path.unlink()


class TestStageTimings:
    """Testes para StageTimings dataclass."""

    def test_stage_timings_to_dict(self):
        """StageTimings.to_dict deve retornar valores arredondados."""
        from src.integration.analysis_orchestrator import StageTimings

        timings = StageTimings(
            transcription_seconds=10.123456,
            acoustic_seconds=5.987654,
            synthesis_seconds=3.456789,
            total_seconds=19.567899,
        )

        result = timings.to_dict()

        assert result["transcription_seconds"] == 10.12
        assert result["acoustic_seconds"] == 5.99
        assert result["synthesis_seconds"] == 3.46
        assert result["total_seconds"] == 19.57


class TestAnalyzeConsultationAsync:
    """Testes para execução assíncrona via job cache."""

    @patch("src.integration.job_cache.get_job_cache")
    @patch("src.integration.audit_log.get_audit_logger")
    @patch("src.audio_processing.whisper_client.transcribe_audio")
    @patch("src.text_processing.gpt4_synthesizer.synthesize_dpp_analysis")
    def test_async_updates_job_cache_on_success(
        self, mock_synthesize, mock_transcribe, mock_audit, mock_cache_func
    ):
        """analyze_consultation_async deve atualizar cache com resultado."""
        import importlib
        import src.integration.analysis_orchestrator as orch_module

        importlib.reload(orch_module)

        from src.integration.job_cache import JobCache, JobStatus
        from src.text_processing.gpt4_synthesizer import (
            AnaliseDPP,
            ComponenteAnalise,
            DPPAssessmentMetadata,
            DPPRiskAssessment,
        )
        from src.audio_processing.whisper_client import TranscriptionResult

        mock_audit_instance = MagicMock()
        mock_audit.return_value = mock_audit_instance

        mock_transcription = TranscriptionResult(
            text="Texto de teste",
            segments=[],
            hesitation_markers=[],
            confidence=0.9,
            language="pt",
            duration=30.0,
        )
        mock_transcribe.return_value = mock_transcription

        mock_assessment = DPPRiskAssessment(
            consultation_id="consul-1",
            analysis_timestamp="2026-02-14T10:00:00Z",
            analise_dpp=AnaliseDPP(
                probabilidade_percentual=50,
                nivel_risco="Moderado",
                indicadores_detectados=[],
                sugerir_alerta=False,
                justificativa_clinica="Teste",
                confianca_analise=0.85,
                componentes_analise=ComponenteAnalise(
                    componente_textual_peso=0.7,
                    componente_acustico_peso=0.3,
                ),
            ),
            metadata=DPPAssessmentMetadata(
                transcription_confidence=0.9,
                acoustic_features_available=True,
                analysis_duration_seconds=5.0,
            ),
        )
        mock_synthesize.return_value = mock_assessment

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")
        mock_cache_func.return_value = cache

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            orch_module.analyze_consultation_async(
                job_id="job-1",
                consultation_id="consul-1",
                audio_path=temp_path,
                correlation_id="corr-1",
            )

            job = cache.get_job("job-1")
            assert job.status == JobStatus.COMPLETED
            assert job.result["analise_dpp"]["probabilidade_percentual"] == 50

        finally:
            temp_path.unlink()

    @patch("src.integration.job_cache.get_job_cache")
    @patch("src.integration.audit_log.get_audit_logger")
    @patch("src.audio_processing.whisper_client.transcribe_audio")
    def test_async_updates_job_cache_on_failure(
        self, mock_transcribe, mock_audit, mock_cache_func
    ):
        """analyze_consultation_async deve atualizar cache com erro."""
        import importlib
        import src.integration.analysis_orchestrator as orch_module

        importlib.reload(orch_module)

        from src.integration.job_cache import JobCache, JobStatus

        mock_audit_instance = MagicMock()
        mock_audit.return_value = mock_audit_instance

        mock_transcribe.side_effect = RuntimeError("Transcription failed")

        cache = JobCache()
        cache.create_job("job-1", "corr-1", "consul-1")
        mock_cache_func.return_value = cache

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            orch_module.analyze_consultation_async(
                job_id="job-1",
                consultation_id="consul-1",
                audio_path=temp_path,
                correlation_id="corr-1",
            )

            job = cache.get_job("job-1")
            assert job.status == JobStatus.FAILED
            assert job.error_code == "ANALYSIS_FAILED"

        finally:
            temp_path.unlink()
