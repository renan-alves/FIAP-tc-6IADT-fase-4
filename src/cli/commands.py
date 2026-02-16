"""CLI command implementations for audio DPP analysis.

This module provides the `analyze` command which validates input and calls
the shared orchestrator.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

from config.logger import get_logger

logger = get_logger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 1
EXIT_AUDIO_ERROR = 2
EXIT_OPENAI_ERROR = 3
EXIT_CONFIG_ERROR = 4


# Supported audio formats per OpenAI Whisper API
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


def validate_file(
    path: Path, max_size_bytes: int = 30 * 1024 * 1024
) -> Tuple[bool, Optional[str]]:
    if not path.exists():
        logger.error("File not found: %s", path)
        return False, "not_found"
    if not path.is_file():
        logger.error("Not a regular file: %s", path)
        return False, "not_file"
    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        logger.error("Unsupported file format: %s", path.suffix)
        return False, "unsupported_format"
    try:
        size = path.stat().st_size
    except OSError as e:
        logger.exception("Could not stat file: %s", e)
        return False, "stat_error"
    if size > max_size_bytes:
        logger.error("File too large: %d bytes (max %d)", size, max_size_bytes)
        return False, "too_large"
    return True, None


def write_output(result: dict, output: Optional[str], fmt: str) -> None:
    from .formatters import format_json, format_text

    if fmt == "json":
        out_str = format_json(result)
    else:
        out_str = format_text(result)

    if output:
        Path(output).write_text(out_str, encoding="utf-8")
        logger.info("Wrote result to %s", output)
    else:
        print(out_str)


def call_orchestrator(
    consultation_id: Optional[str], audio_path: Path, no_cleanup: bool
):
    try:
        from src.integration.analysis_orchestrator import analyze_consultation
    except Exception as e:
        logger.exception("Could not import orchestrator: %s", e)
        return None, EXIT_CONFIG_ERROR

    try:
        # Orchestrator is expected to be synchronous for CLI usage
        result = analyze_consultation(
            consultation_id, audio_path, no_cleanup=no_cleanup
        )
        return result, EXIT_SUCCESS
    except Exception:
        logger.exception("Processing failed")
        return None, EXIT_OPENAI_ERROR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.cli", description="Audio DPP analysis CLI"
    )
    sub = parser.add_subparsers(dest="command")

    an = sub.add_parser("analyze", help="Analyze an audio consultation file")
    an.add_argument("audio_file", type=str)
    an.add_argument("--consultation-id", type=str, default=None)
    an.add_argument("--output", "-o", type=str, default=None)
    an.add_argument("--format", choices=("json", "text"), default="json")
    an.add_argument("--verbose", action="store_true")
    an.add_argument("--no-cleanup", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        audio_path = Path(args.audio_file)
        valid, reason = validate_file(audio_path)
        if not valid:
            if reason == "too_large":
                return EXIT_AUDIO_ERROR
            return EXIT_INVALID_INPUT

        result, code = call_orchestrator(
            args.consultation_id, audio_path, args.no_cleanup
        )
        if code != EXIT_SUCCESS:
            return code

        write_output(
            {"consultation_id": args.consultation_id, "result": result},
            args.output,
            args.format,
        )
        return EXIT_SUCCESS

    parser.print_help()
    return EXIT_INVALID_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
