"""Testes para o cliente OpenAI Whisper."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestHesitationMarker:
    """Testes para a classe HesitationMarker."""

    def test_to_dict_pause(self):
        from src.audio_processing.whisper_client import HesitationMarker

        marker = HesitationMarker(
            type="pause",
            start_time=5.0,
            end_time=7.5,
            text=None,
            duration=2.5,
        )
        result = marker.to_dict()

        assert result["type"] == "pause"
        assert result["start_time"] == 5.0
        assert result["end_time"] == 7.5
        assert result["text"] is None
        assert result["duration"] == 2.5

    def test_to_dict_filler_word(self):
        from src.audio_processing.whisper_client import HesitationMarker

        marker = HesitationMarker(
            type="filler_word",
            start_time=10.0,
            end_time=10.5,
            text="então",
            duration=0.5,
        )
        result = marker.to_dict()

        assert result["type"] == "filler_word"
        assert result["text"] == "então"


class TestTranscriptionSegment:
    """Testes para a classe TranscriptionSegment."""

    def test_to_dict(self):
        from src.audio_processing.whisper_client import TranscriptionSegment

        segment = TranscriptionSegment(
            id=0,
            start=0.0,
            end=5.0,
            text="Olá, como você está?",
        )
        result = segment.to_dict()

        assert result["id"] == 0
        assert result["start"] == 0.0
        assert result["end"] == 5.0
        assert result["text"] == "Olá, como você está?"


class TestTranscriptionResult:
    """Testes para a classe TranscriptionResult."""

    def test_to_dict_empty(self):
        from src.audio_processing.whisper_client import TranscriptionResult

        result = TranscriptionResult(text="Teste")
        d = result.to_dict()

        assert d["text"] == "Teste"
        assert d["segments"] == []
        assert d["hesitation_markers"] == []
        assert d["confidence"] == 0.0
        assert d["language"] == "pt"

    def test_to_dict_with_segments(self):
        from src.audio_processing.whisper_client import (
            HesitationMarker,
            TranscriptionResult,
            TranscriptionSegment,
        )

        result = TranscriptionResult(
            text="Olá, como você está?",
            segments=[
                TranscriptionSegment(
                    id=0, start=0.0, end=5.0, text="Olá, como você está?"
                )
            ],
            hesitation_markers=[
                HesitationMarker(
                    type="pause", start_time=5.0, end_time=7.0, duration=2.0
                )
            ],
            confidence=0.95,
            language="pt",
            duration=7.0,
        )
        d = result.to_dict()

        assert len(d["segments"]) == 1
        assert len(d["hesitation_markers"]) == 1
        assert d["confidence"] == 0.95
        assert d["duration"] == 7.0


class TestFillerWordDetection:
    """Testes para detecção de filler words."""

    def test_detect_single_filler(self):
        from src.audio_processing.whisper_client import _detect_filler_words

        markers = _detect_filler_words("Então, eu acho que sim", 0.0, 5.0)
        filler_texts = [m.text for m in markers]

        assert "então" in filler_texts

    def test_detect_multiple_fillers(self):
        from src.audio_processing.whisper_client import _detect_filler_words

        markers = _detect_filler_words("Então, tipo, sabe, eu não sei", 0.0, 10.0)
        filler_texts = [m.text for m in markers]

        assert "então" in filler_texts
        assert "tipo" in filler_texts
        assert "sabe" in filler_texts

    def test_no_fillers(self):
        from src.audio_processing.whisper_client import _detect_filler_words

        markers = _detect_filler_words("Eu estou ótimo obrigado", 0.0, 5.0)

        assert len(markers) == 0

    def test_filler_timestamps(self):
        from src.audio_processing.whisper_client import _detect_filler_words

        markers = _detect_filler_words("Então sim", 10.0, 12.0)

        assert len(markers) == 1
        assert markers[0].start_time >= 10.0
        assert markers[0].end_time <= 12.0


class TestPauseDetection:
    """Testes para detecção de pausas."""

    def test_detect_long_pause(self):
        from src.audio_processing.whisper_client import (
            TranscriptionSegment,
            _detect_pauses,
        )

        segments = [
            TranscriptionSegment(id=0, start=0.0, end=5.0, text="Primeiro"),
            TranscriptionSegment(id=1, start=7.0, end=10.0, text="Segundo"),  # 2s gap
        ]
        markers = _detect_pauses(segments)

        assert len(markers) == 1
        assert markers[0].type == "pause"
        assert markers[0].duration == 2.0

    def test_no_long_pause(self):
        from src.audio_processing.whisper_client import (
            TranscriptionSegment,
            _detect_pauses,
        )

        segments = [
            TranscriptionSegment(id=0, start=0.0, end=5.0, text="Primeiro"),
            TranscriptionSegment(id=1, start=5.5, end=10.0, text="Segundo"),  # 0.5s gap
        ]
        markers = _detect_pauses(segments)

        assert len(markers) == 0

    def test_multiple_pauses(self):
        from src.audio_processing.whisper_client import (
            TranscriptionSegment,
            _detect_pauses,
        )

        segments = [
            TranscriptionSegment(id=0, start=0.0, end=2.0, text="Um"),
            TranscriptionSegment(id=1, start=5.0, end=7.0, text="Dois"),  # 3s gap
            TranscriptionSegment(id=2, start=10.0, end=12.0, text="Três"),  # 3s gap
        ]
        markers = _detect_pauses(segments)

        assert len(markers) == 2


class TestTempFileManagement:
    """Testes para gerenciamento de arquivos temporários."""

    def test_save_audio_to_temp(self):
        from src.audio_processing.whisper_client import save_audio_to_temp

        audio_bytes = b"fake audio content"
        path = save_audio_to_temp(audio_bytes, suffix=".wav")

        try:
            assert path.exists()
            assert path.suffix == ".wav"
            assert path.read_bytes() == audio_bytes
        finally:
            path.unlink()

    def test_cleanup_temp_audio(self):
        from src.audio_processing.whisper_client import cleanup_temp_audio

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(tmp_path)

        cleanup_temp_audio(path)

        assert not path.exists()

    def test_cleanup_nonexistent_file(self):
        from src.audio_processing.whisper_client import cleanup_temp_audio

        path = Path("/tmp/nonexistent_audio_file_12345.wav")
        # Should not raise
        cleanup_temp_audio(path)


class TestTranscribeAudioValidation:
    """Testes de validação para transcrição de áudio."""

    def test_file_not_found(self):
        from src.audio_processing.whisper_client import transcribe_audio

        with pytest.raises(FileNotFoundError) as exc_info:
            transcribe_audio(Path("/nonexistent/audio.wav"))

        assert "não encontrado" in str(exc_info.value)

    def test_missing_api_key(self):
        from src.audio_processing.whisper_client import transcribe_audio

        # Create temp audio file
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(tmp_path)

        try:
            # Mock the openai module to be available but no API key
            mock_openai = mock.MagicMock()
            with mock.patch.dict("sys.modules", {"openai": mock_openai}):
                with mock.patch.dict(os.environ, {}, clear=True):
                    # Ensure OPENAI_API_KEY is not set
                    os.environ.pop("OPENAI_API_KEY", None)
                    with pytest.raises(ValueError) as exc_info:
                        transcribe_audio(path)

                    assert "OPENAI_API_KEY" in str(exc_info.value)
        finally:
            path.unlink()


class TestTranscribeAudioMocked:
    """Testes com mock da API OpenAI."""

    @pytest.fixture
    def mock_openai_response(self):
        """Cria resposta mock da API Whisper."""

        class MockSegment:
            def __init__(self, id, start, end, text, avg_logprob=-0.3):
                self.id = id
                self.start = start
                self.end = end
                self.text = text
                self.avg_logprob = avg_logprob

        class MockResponse:
            def __init__(self):
                self.text = "Olá, então eu estou me sentindo um pouco cansada."
                self.duration = 10.0
                self.segments = [
                    MockSegment(0, 0.0, 3.0, "Olá,"),
                    MockSegment(
                        1, 5.0, 10.0, "então eu estou me sentindo um pouco cansada."
                    ),
                ]

        return MockResponse()

    def test_transcribe_success(self, mock_openai_response):
        from src.audio_processing.whisper_client import transcribe_audio

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(tmp_path)

        try:
            # Create mock OpenAI module and client
            mock_client = mock.MagicMock()
            mock_client.audio.transcriptions.create.return_value = mock_openai_response

            mock_openai_module = mock.MagicMock()
            mock_openai_module.OpenAI.return_value = mock_client

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with mock.patch.dict("sys.modules", {"openai": mock_openai_module}):
                    result = transcribe_audio(path)

                    assert result.text == mock_openai_response.text
                    assert len(result.segments) == 2
                    assert result.duration == 10.0
                    # Should find pause between segments (3.0 to 5.0 = 2s pause)
                    pause_markers = [
                        m for m in result.hesitation_markers if m.type == "pause"
                    ]
                    assert len(pause_markers) == 1
                    # Should find filler word "então"
                    filler_markers = [
                        m for m in result.hesitation_markers if m.type == "filler_word"
                    ]
                    assert any(m.text == "então" for m in filler_markers)
        finally:
            path.unlink()

    def test_transcribe_retry_on_failure(self, mock_openai_response):
        from src.audio_processing.whisper_client import transcribe_audio

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(tmp_path)

        try:
            # Create mock OpenAI module and client
            mock_client = mock.MagicMock()
            # Fail twice, succeed on third attempt
            mock_client.audio.transcriptions.create.side_effect = [
                Exception("API Error 1"),
                Exception("API Error 2"),
                mock_openai_response,
            ]

            mock_openai_module = mock.MagicMock()
            mock_openai_module.OpenAI.return_value = mock_client

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with mock.patch.dict("sys.modules", {"openai": mock_openai_module}):
                    with mock.patch("time.sleep"):  # Skip actual sleep
                        result = transcribe_audio(path)

                    assert result.text == mock_openai_response.text
                    assert mock_client.audio.transcriptions.create.call_count == 3
        finally:
            path.unlink()

    def test_transcribe_all_retries_fail(self):
        from src.audio_processing.whisper_client import transcribe_audio

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(tmp_path)

        try:
            # Create mock OpenAI module and client
            mock_client = mock.MagicMock()
            mock_client.audio.transcriptions.create.side_effect = Exception("API Error")

            mock_openai_module = mock.MagicMock()
            mock_openai_module.OpenAI.return_value = mock_client

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with mock.patch.dict("sys.modules", {"openai": mock_openai_module}):
                    with mock.patch("time.sleep"):
                        with pytest.raises(RuntimeError) as exc_info:
                            transcribe_audio(path)

                        assert "falhou após 3 tentativas" in str(exc_info.value)
                        assert mock_client.audio.transcriptions.create.call_count == 3
        finally:
            path.unlink()


class TestConfidenceCalculation:
    """Testes para cálculo de confiança."""

    def test_confidence_from_logprobs(self):
        from src.audio_processing.whisper_client import _calculate_confidence

        class MockSegment:
            avg_logprob = -0.2

        class MockResponse:
            segments = [MockSegment(), MockSegment()]

        result = _calculate_confidence(MockResponse())

        assert 0.0 <= result <= 1.0
        assert result == 0.8  # 1.0 + (-0.2) = 0.8

    def test_confidence_no_segments(self):
        from src.audio_processing.whisper_client import _calculate_confidence

        class MockResponse:
            pass

        result = _calculate_confidence(MockResponse())

        assert result == 0.8  # Default value

    def test_confidence_caps_at_bounds(self):
        from src.audio_processing.whisper_client import _calculate_confidence

        class MockSegmentHigh:
            avg_logprob = 0.5  # Would be > 1.0

        class MockSegmentLow:
            avg_logprob = -2.0  # Would be < 0.0

        class MockResponseHigh:
            segments = [MockSegmentHigh()]

        class MockResponseLow:
            segments = [MockSegmentLow()]

        high_result = _calculate_confidence(MockResponseHigh())
        low_result = _calculate_confidence(MockResponseLow())

        assert high_result == 1.0
        assert low_result == 0.0
