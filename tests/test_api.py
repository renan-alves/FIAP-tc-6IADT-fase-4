"""Testes para a API REST FastAPI de análise DPP.

Cobertura:
- Endpoints: POST /analyze, GET /analyze/{job_id}, GET /health
- Validação de request/response
- Error handling e envelopes de erro
- Job polling e status
- Autenticação bearer token
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest

# Test client requires httpx installed
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from src.api.app import (
    SUPPORTED_AUDIO_FORMATS,
    MAX_FILE_SIZE,
    app,
    _status_to_error_code,
)
from src.api.models import (
    AnalyzeAccepted,
    ErrorEnvelope,
    HealthResponse,
    RiskLevel,
    JobStatusEnum,
)


# --- Fixtures ---


@pytest.fixture(autouse=True)
def clear_auth_token():
    """Clear API_BEARER_TOKEN for all tests by default."""
    orig = os.environ.pop("API_BEARER_TOKEN", None)
    yield
    if orig:
        os.environ["API_BEARER_TOKEN"] = orig


@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_wav_bytes():
    """Create minimal WAV file bytes for testing."""
    # Minimal valid WAV header (44 bytes) + 1 second of silence
    wav_header = bytes(
        [
            0x52,
            0x49,
            0x46,
            0x46,  # "RIFF"
            0x24,
            0x00,
            0x00,
            0x00,  # File size - 8
            0x57,
            0x41,
            0x56,
            0x45,  # "WAVE"
            0x66,
            0x6D,
            0x74,
            0x20,  # "fmt "
            0x10,
            0x00,
            0x00,
            0x00,  # Subchunk1 size (16)
            0x01,
            0x00,  # Audio format (1 = PCM)
            0x01,
            0x00,  # Num channels (1)
            0x44,
            0xAC,
            0x00,
            0x00,  # Sample rate (44100)
            0x88,
            0x58,
            0x01,
            0x00,  # Byte rate
            0x02,
            0x00,  # Block align
            0x10,
            0x00,  # Bits per sample (16)
            0x64,
            0x61,
            0x74,
            0x61,  # "data"
            0x00,
            0x00,
            0x00,
            0x00,  # Subchunk2 size
        ]
    )
    return wav_header + b"\x00" * 1000


@pytest.fixture
def mock_job_cache():
    """Mock job cache for testing."""
    with mock.patch("src.api.app._get_job_cache") as mock_get:
        mock_cache = mock.MagicMock()
        mock_get.return_value = mock_cache
        yield mock_cache


# --- Test Models ---


class TestModels:
    """Test Pydantic models validation."""

    def test_analyze_accepted_valid(self):
        """Test AnalyzeAccepted with valid data."""
        model = AnalyzeAccepted(
            job_id="123e4567-e89b-12d3-a456-426614174000",
            correlation_id="987fcdeb-51a2-43e7-b8c9-123456789abc",
        )
        assert model.job_id == "123e4567-e89b-12d3-a456-426614174000"
        assert model.correlation_id == "987fcdeb-51a2-43e7-b8c9-123456789abc"

    def test_error_envelope_with_retry(self):
        """Test ErrorEnvelope with retry_after."""
        error = ErrorEnvelope(
            error_code="SERVICE_UNAVAILABLE",
            message="OpenAI API unavailable",
            retry_after=30,
        )
        assert error.error_code == "SERVICE_UNAVAILABLE"
        assert error.retry_after == 30

    def test_error_envelope_without_retry(self):
        """Test ErrorEnvelope without retry_after."""
        error = ErrorEnvelope(
            error_code="BAD_REQUEST",
            message="Invalid input",
        )
        assert error.retry_after is None

    def test_health_response(self):
        """Test HealthResponse model."""
        health = HealthResponse(
            status="ok",
            timestamp="2026-02-15T10:00:00Z",
            version="1.0.0",
            dependencies=[],
        )
        assert health.status == "ok"
        assert health.version == "1.0.0"

    def test_risk_level_enum(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.BAIXO.value == "Baixo"
        assert RiskLevel.MODERADO.value == "Moderado"
        assert RiskLevel.ALTO.value == "Alto"
        assert RiskLevel.CRITICO.value == "Crítico"

    def test_job_status_enum(self):
        """Test JobStatusEnum values."""
        assert JobStatusEnum.PENDING.value == "pending"
        assert JobStatusEnum.PROCESSING.value == "processing"
        assert JobStatusEnum.COMPLETED.value == "completed"
        assert JobStatusEnum.FAILED.value == "failed"


class TestStatusToErrorCode:
    """Test HTTP status to error code mapping."""

    def test_known_status_codes(self):
        """Test mapping for known status codes."""
        assert _status_to_error_code(400) == "BAD_REQUEST"
        assert _status_to_error_code(401) == "UNAUTHORIZED"
        assert _status_to_error_code(403) == "FORBIDDEN"
        assert _status_to_error_code(404) == "NOT_FOUND"
        assert _status_to_error_code(413) == "FILE_TOO_LARGE"
        assert _status_to_error_code(415) == "UNSUPPORTED_MEDIA_TYPE"
        assert _status_to_error_code(500) == "INTERNAL_ERROR"
        assert _status_to_error_code(503) == "SERVICE_UNAVAILABLE"

    def test_unknown_status_code(self):
        """Test mapping for unknown status code."""
        assert _status_to_error_code(418) == "UNKNOWN_ERROR"


# --- Test Health Endpoint ---


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Test health check returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Test health response has expected fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "dependencies" in data
        assert isinstance(data["dependencies"], list)

    def test_health_status_ok_with_api_key(self, client):
        """Test health status is ok when OPENAI_API_KEY is set."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            response = client.get("/health")
            data = response.json()

            assert data["status"] == "ok"
            openai_dep = next(d for d in data["dependencies"] if d["name"] == "openai")
            assert openai_dep["status"] == "ok"

    def test_health_status_degraded_without_api_key(self, client):
        """Test health status is degraded when OPENAI_API_KEY is not set."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            # Need to temporarily clear the key
            orig_key = os.environ.pop("OPENAI_API_KEY", None)
            try:
                response = client.get("/health")
                data = response.json()

                assert data["status"] == "degraded"
                openai_dep = next(
                    d for d in data["dependencies"] if d["name"] == "openai"
                )
                assert openai_dep["status"] == "unavailable"
            finally:
                if orig_key:
                    os.environ["OPENAI_API_KEY"] = orig_key


# --- Test POST /analyze Endpoint ---


class TestPostAnalyzeEndpoint:
    """Test POST /analyze endpoint."""

    def test_missing_audio_file_returns_400(self, client):
        """Test missing audio_file returns 400 Bad Request."""
        response = client.post(
            "/analyze",
            data={"consultation_id": "test-123"},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "BAD_REQUEST"
        assert "audio_file" in data["message"].lower()

    def test_unsupported_format_returns_415(self, client):
        """Test unsupported audio format returns 415."""
        response = client.post(
            "/analyze",
            files={"audio_file": ("test.txt", b"not audio", "text/plain")},
        )
        assert response.status_code == 415
        data = response.json()
        assert data["error_code"] == "UNSUPPORTED_MEDIA_TYPE"

    def test_valid_wav_returns_202(self, client, sample_wav_bytes, mock_job_cache):
        """Test valid WAV file returns 202 Accepted."""
        mock_job_cache.create_job.return_value = mock.MagicMock(
            job_id="test-job-id",
            correlation_id="test-correlation-id",
        )

        response = client.post(
            "/analyze",
            files={"audio_file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"consultation_id": "consul-123"},
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert "correlation_id" in data

    def test_valid_mp3_returns_202(self, client, mock_job_cache):
        """Test valid MP3 file returns 202 Accepted."""
        mock_job_cache.create_job.return_value = mock.MagicMock()

        # Minimal MP3 header
        mp3_bytes = b"\xff\xfb\x90\x00" + b"\x00" * 100

        response = client.post(
            "/analyze",
            files={"audio_file": ("test.mp3", mp3_bytes, "audio/mpeg")},
        )

        assert response.status_code == 202

    def test_all_supported_formats_accepted(self, client, mock_job_cache):
        """Test all supported audio formats are accepted."""
        mock_job_cache.create_job.return_value = mock.MagicMock()

        for ext in SUPPORTED_AUDIO_FORMATS:
            response = client.post(
                "/analyze",
                files={
                    "audio_file": (
                        f"test{ext}",
                        b"\x00" * 100,
                        "application/octet-stream",
                    )
                },
            )
            assert response.status_code == 202, f"Format {ext} should be accepted"


# --- Test GET /analyze/{job_id} Endpoint ---


class TestGetAnalyzeEndpoint:
    """Test GET /analyze/{job_id} endpoint."""

    def test_job_not_found_returns_404(self, client, mock_job_cache):
        """Test non-existent job returns 404."""
        mock_job_cache.get_job.return_value = None

        response = client.get("/analyze/non-existent-job-id")

        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == "NOT_FOUND"

    def test_job_processing_returns_202(self, client, mock_job_cache):
        """Test processing job returns 202."""
        from src.integration.job_cache import JobStatus

        mock_job = mock.MagicMock()
        mock_job.status = JobStatus.PROCESSING
        mock_job_cache.get_job.return_value = mock_job

        response = client.get("/analyze/test-job-id")

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "processing"

    def test_job_pending_returns_202(self, client, mock_job_cache):
        """Test pending job returns 202."""
        from src.integration.job_cache import JobStatus

        mock_job = mock.MagicMock()
        mock_job.status = JobStatus.PENDING
        mock_job_cache.get_job.return_value = mock_job

        response = client.get("/analyze/test-job-id")

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"

    def test_job_completed_returns_200(self, client, mock_job_cache):
        """Test completed job returns 200 with result."""
        from src.integration.job_cache import JobStatus

        mock_result = {
            "consultation_id": "test-consul",
            "analysis_timestamp": "2026-02-15T10:00:00Z",
            "analise_dpp": {
                "probabilidade_percentual": 45,
                "nivel_risco": "Moderado",
            },
            "metadata": {},
        }

        mock_job = mock.MagicMock()
        mock_job.status = JobStatus.COMPLETED
        mock_job.result = mock_result
        mock_job_cache.get_job.return_value = mock_job

        response = client.get("/analyze/test-job-id")

        assert response.status_code == 200
        data = response.json()
        assert data["consultation_id"] == "test-consul"
        assert data["analise_dpp"]["probabilidade_percentual"] == 45

    def test_job_failed_returns_500(self, client, mock_job_cache):
        """Test failed job returns 500 with error."""
        from src.integration.job_cache import JobStatus

        mock_job = mock.MagicMock()
        mock_job.status = JobStatus.FAILED
        mock_job.error_code = "TRANSCRIPTION_FAILED"
        mock_job.error_message = "OpenAI API error"
        mock_job_cache.get_job.return_value = mock_job

        response = client.get("/analyze/test-job-id")

        assert response.status_code == 500
        data = response.json()
        assert data["error_code"] == "TRANSCRIPTION_FAILED"
        assert "OpenAI" in data["message"]


# --- Test Authentication ---


class TestAuthentication:
    """Test bearer token authentication."""

    def test_no_auth_when_token_not_configured(self, client):
        """Test requests pass when API_BEARER_TOKEN not set."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("API_BEARER_TOKEN", None)
            response = client.get("/health")
            assert response.status_code == 200

    def test_missing_token_returns_401(self, client):
        """Test missing token returns 401 when token is configured."""
        with mock.patch.dict(os.environ, {"API_BEARER_TOKEN": "secret-token"}):
            response = client.get("/health")
            # Health doesn't require auth, test with analyze
            response = client.post(
                "/analyze",
                data={},
            )
            assert response.status_code == 401

    def test_invalid_token_returns_403(self, client):
        """Test invalid token returns 403."""
        with mock.patch.dict(os.environ, {"API_BEARER_TOKEN": "secret-token"}):
            response = client.post(
                "/analyze",
                data={},
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert response.status_code == 403

    def test_valid_token_accepted(self, client, mock_job_cache):
        """Test valid token is accepted."""
        mock_job_cache.create_job.return_value = mock.MagicMock()

        with mock.patch.dict(os.environ, {"API_BEARER_TOKEN": "secret-token"}):
            response = client.post(
                "/analyze",
                files={"audio_file": ("test.wav", b"\x00" * 100, "audio/wav")},
                headers={"Authorization": "Bearer secret-token"},
            )
            assert response.status_code == 202


# --- Test File Size Validation ---


class TestFileSizeValidation:
    """Test file size validation."""

    def test_large_file_returns_413(self, client):
        """Test file exceeding MAX_FILE_SIZE returns 413."""
        # Create data slightly larger than max
        large_data = b"\x00" * (MAX_FILE_SIZE + 1000)

        response = client.post(
            "/analyze",
            files={"audio_file": ("test.wav", large_data, "audio/wav")},
        )

        assert response.status_code == 413
        data = response.json()
        assert data["error_code"] == "FILE_TOO_LARGE"


# --- Test Supported Formats Constant ---


class TestSupportedFormats:
    """Test SUPPORTED_AUDIO_FORMATS constant."""

    def test_wav_supported(self):
        """Test WAV format is supported."""
        assert ".wav" in SUPPORTED_AUDIO_FORMATS

    def test_mp3_supported(self):
        """Test MP3 format is supported."""
        assert ".mp3" in SUPPORTED_AUDIO_FORMATS

    def test_mp4_supported(self):
        """Test MP4 format is supported."""
        assert ".mp4" in SUPPORTED_AUDIO_FORMATS

    def test_flac_supported(self):
        """Test FLAC format is supported."""
        assert ".flac" in SUPPORTED_AUDIO_FORMATS

    def test_case_insensitive_extensions(self):
        """Test that all extensions are lowercase."""
        for ext in SUPPORTED_AUDIO_FORMATS:
            assert ext == ext.lower()
            assert ext.startswith(".")
