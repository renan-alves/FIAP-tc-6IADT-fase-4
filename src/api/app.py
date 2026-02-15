"""FastAPI REST API for DPP Audio Analysis.

Provides async job-based audio analysis with polling endpoints.
Implements GOAL-005 per implementation-plan.md.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config.logger import get_logger

from . import models
from .models import (
    AnalyzeAccepted,
    DependencyStatus,
    ErrorEnvelope,
    HealthResponse,
)

logger = get_logger(__name__)

# Supported audio formats (aligned with CLI and Whisper API)
SUPPORTED_AUDIO_FORMATS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
    ".mpeg",
    ".mpga",
    ".oga",
}

# Maximum file size (30MB)
MAX_FILE_SIZE = 30 * 1024 * 1024

# API version
API_VERSION = "1.0.0"


# --- Lifespan Context Manager ---

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown events."""
    logger.info("Starting DPP Audio Analysis API v%s", API_VERSION)
    cleanup_task = asyncio.create_task(_cleanup_expired_jobs())
    yield
    logger.info("Shutting down DPP Audio Analysis API")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# --- FastAPI App ---

app = FastAPI(
    title="DPP Audio Analysis API",
    description="API para análise de depressão pós-parto via áudio de consultas clínicas",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# --- Exception Handlers ---


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors with standard error envelope."""
    logger.warning("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorEnvelope(
            error_code="VALIDATION_ERROR",
            message=f"Request validation failed: {exc.errors()}",
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Wrap HTTPException in standard error envelope."""
    error_code = _status_to_error_code(exc.status_code)
    retry_after = None
    if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        retry_after = 30

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorEnvelope(
            error_code=error_code,
            message=str(exc.detail),
            retry_after=retry_after,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception("Unexpected error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorEnvelope(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred",
        ).model_dump(),
    )


def _status_to_error_code(status_code: int) -> str:
    """Map HTTP status code to error code."""
    mapping = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        413: "FILE_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    return mapping.get(status_code, "UNKNOWN_ERROR")


# --- Authentication ---


def _require_bearer_token(request: Request) -> None:
    """Validate optional bearer token if configured."""
    token = os.getenv("API_BEARER_TOKEN")
    if not token:
        return None
    auth = request.headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    provided = auth.split(" ", 1)[1].strip()
    if provided != token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid bearer token",
        )


# --- Job Cache Integration ---


def _get_job_cache():
    """Get the global job cache instance."""
    from src.integration.job_cache import get_job_cache

    return get_job_cache()


# --- Helper Functions ---


async def _save_upload_to_temp(upload, max_size: int = MAX_FILE_SIZE) -> Path:
    """Save uploaded file to temp directory with size validation."""
    suffix = Path(upload.filename).suffix.lower() if upload.filename else ""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(tmp_path)
    size = 0
    too_large = False

    try:
        with path.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 64)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    too_large = True
                    break
                f.write(chunk)
    finally:
        await upload.close()

    if too_large:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB",
        )

    logger.debug("Saved upload to %s (%d bytes)", path, size)
    return path


async def _run_orchestrator_async(
    job_id: str,
    consultation_id: Optional[str],
    audio_path: Path,
) -> None:
    """Run the orchestrator in a thread pool executor."""
    from src.integration.analysis_orchestrator import analyze_consultation
    from src.integration.job_cache import JobStatus

    job_cache = _get_job_cache()

    # Update status to processing
    job_cache.update_status(job_id, JobStatus.PROCESSING)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, analyze_consultation, consultation_id, audio_path, False
        )

        # Complete job with result
        job_cache.complete_job(job_id, result)
        logger.info("Job %s completed successfully", job_id)

    except FileNotFoundError as e:
        job_cache.fail_job(job_id, "AUDIO_NOT_FOUND", str(e))
        logger.error("Job %s failed: %s", job_id, e)

    except ValueError as e:
        job_cache.fail_job(job_id, "INVALID_INPUT", str(e))
        logger.error("Job %s failed: %s", job_id, e)

    except RuntimeError as e:
        error_msg = str(e)
        if "OPENAI" in error_msg.upper() or "transcri" in error_msg.lower():
            job_cache.fail_job(job_id, "TRANSCRIPTION_FAILED", error_msg)
        elif "synthe" in error_msg.lower():
            job_cache.fail_job(job_id, "SYNTHESIS_FAILED", error_msg)
        else:
            job_cache.fail_job(job_id, "PROCESSING_FAILED", error_msg)
        logger.error("Job %s failed: %s", job_id, e)

    except Exception as e:
        job_cache.fail_job(job_id, "UNEXPECTED_ERROR", str(e))
        logger.exception("Job %s unexpected error: %s", job_id, e)

    finally:
        # Cleanup temp file
        try:
            if audio_path.exists():
                audio_path.unlink()
                logger.debug("Cleaned up temp file: %s", audio_path)
        except Exception as cleanup_err:
            logger.warning("Failed to cleanup temp file: %s", cleanup_err)


# --- API Endpoints ---


@app.post(
    "/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AnalyzeAccepted,
    responses={
        202: {"description": "Job accepted for processing"},
        400: {"model": ErrorEnvelope, "description": "Bad request"},
        401: {"model": ErrorEnvelope, "description": "Unauthorized"},
        413: {"model": ErrorEnvelope, "description": "File too large"},
        415: {"model": ErrorEnvelope, "description": "Unsupported media type"},
        503: {"model": ErrorEnvelope, "description": "Service unavailable"},
    },
    summary="Submit audio for DPP analysis",
    description="Upload an audio file for postpartum depression risk analysis. Returns a job_id for polling.",
)
async def post_analyze(
    request: Request,
    auth: None = Depends(_require_bearer_token),
) -> JSONResponse:
    """Accept audio file and start async analysis job."""
    # Parse multipart form
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse form data: {e}",
        )

    audio_file = form.get("audio_file")
    consultation_id = form.get("consultation_id")

    if audio_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required field: audio_file",
        )

    # Validate filename and extension
    filename = getattr(audio_file, "filename", None)
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload: missing filename",
        )

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format: {ext}. Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}",
        )

    # Generate IDs
    job_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())

    # Save upload to temp file
    try:
        audio_path = await _save_upload_to_temp(audio_file)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to save upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded file",
        )

    # Create job in cache
    job_cache = _get_job_cache()
    job_cache.create_job(
        job_id=job_id,
        correlation_id=correlation_id,
        consultation_id=consultation_id or f"api-{job_id[:8]}",
        audio_path=str(audio_path),
    )

    logger.info(
        "Created job %s for consultation %s, correlation %s",
        job_id,
        consultation_id,
        correlation_id,
    )

    # Spawn background task
    asyncio.create_task(_run_orchestrator_async(job_id, consultation_id, audio_path))

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"job_id": job_id, "correlation_id": correlation_id},
    )


@app.get(
    "/analyze/{job_id}",
    responses={
        200: {"description": "Analysis completed"},
        202: {"model": models.AnalyzeStatusProcessing, "description": "Still processing"},
        404: {"model": ErrorEnvelope, "description": "Job not found"},
        500: {"model": ErrorEnvelope, "description": "Processing failed"},
    },
    summary="Get analysis job status",
    description="Poll for analysis result by job_id. Returns 202 if processing, 200 if complete.",
)
async def get_analyze(
    job_id: str,
    request: Request,
    auth: None = Depends(_require_bearer_token),
) -> JSONResponse:
    """Get job status and result if complete."""
    job_cache = _get_job_cache()
    job = job_cache.get_job(job_id)

    if not job:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorEnvelope(
                error_code="NOT_FOUND",
                message=f"Job not found or expired: {job_id}",
            ).model_dump(),
        )

    # Check status
    if job.status.value in ("pending", "processing"):
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": job.status.value},
        )

    if job.status.value == "completed":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=job.result,
        )

    # Failed
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorEnvelope(
            error_code=job.error_code or "PROCESSING_FAILED",
            message=job.error_message or "Analysis failed",
        ).model_dump(),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and dependency status.",
)
async def health() -> HealthResponse:
    """Health check endpoint with dependency status."""
    dependencies = []
    overall_status = "ok"

    # Check OpenAI API
    openai_status = await _check_openai_health()
    dependencies.append(openai_status)
    if openai_status.status != "ok":
        overall_status = "degraded"

    # Check job cache
    cache_status = _check_cache_health()
    dependencies.append(cache_status)

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return HealthResponse(
        status=overall_status,
        timestamp=timestamp,
        version=API_VERSION,
        dependencies=dependencies,
    )


async def _check_openai_health() -> DependencyStatus:
    """Check OpenAI API availability."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return DependencyStatus(
            name="openai",
            status="unavailable",
            message="OPENAI_API_KEY not configured",
        )

    # Simple check - just verify the key is set
    # Full health check would call the API but that adds latency
    return DependencyStatus(
        name="openai",
        status="ok",
        message="API key configured",
    )


def _check_cache_health() -> DependencyStatus:
    """Check job cache health."""
    try:
        job_cache = _get_job_cache()
        stats = job_cache.get_stats()
        return DependencyStatus(
            name="job_cache",
            status="ok",
            message=f"Active jobs: {stats['total_jobs']}",
        )
    except Exception as e:
        return DependencyStatus(
            name="job_cache",
            status="degraded",
            message=str(e),
        )


# --- Background Tasks ---


async def _cleanup_expired_jobs():
    """Periodically clean up expired jobs and temp files."""
    while True:
        await asyncio.sleep(60 * 10)  # Every 10 minutes
        try:
            job_cache = _get_job_cache()
            # Cleanup is handled internally by job_cache on access
            stats = job_cache.get_stats()
            logger.debug("Job cache stats: %s", stats)
        except Exception as e:
            logger.warning("Cleanup task error: %s", e)
