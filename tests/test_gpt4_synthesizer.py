"""Testes para o sintetizador GPT-4 de risco de DPP."""

import json
import os
from unittest import mock

import pytest


class TestDataclasses:
    """Testes para as dataclasses de avaliação DPP."""

    def test_componente_analise_to_dict(self):
        from src.text_processing.gpt4_synthesizer import ComponenteAnalise

        comp = ComponenteAnalise(
            componente_textual_peso=0.6,
            componente_acustico_peso=0.4,
        )
        result = comp.to_dict()

        assert result["componente_textual_peso"] == 0.6
        assert result["componente_acustico_peso"] == 0.4

    def test_analise_dpp_to_dict(self):
        from src.text_processing.gpt4_synthesizer import AnaliseDPP, ComponenteAnalise

        analise = AnaliseDPP(
            probabilidade_percentual=75,
            nivel_risco="Alto",
            indicadores_detectados=["Fadiga extrema", "Isolamento social"],
            sugerir_alerta=True,
            justificativa_clinica="Múltiplos indicadores presentes.",
            confianca_analise=0.85,
            componentes_analise=ComponenteAnalise(0.6, 0.4),
        )
        result = analise.to_dict()

        assert result["probabilidade_percentual"] == 75
        assert result["nivel_risco"] == "Alto"
        assert len(result["indicadores_detectados"]) == 2
        assert result["sugerir_alerta"] is True
        assert result["confianca_analise"] == 0.85

    def test_dpp_assessment_metadata_to_dict(self):
        from src.text_processing.gpt4_synthesizer import DPPAssessmentMetadata

        meta = DPPAssessmentMetadata(
            transcription_confidence=0.92,
            acoustic_features_available=True,
            analysis_duration_seconds=5.5,
        )
        result = meta.to_dict()

        assert result["transcription_confidence"] == 0.92
        assert result["acoustic_features_available"] is True
        assert result["analysis_duration_seconds"] == 5.5
        assert "quality_warning" not in result

    def test_dpp_assessment_metadata_with_warning(self):
        from src.text_processing.gpt4_synthesizer import DPPAssessmentMetadata

        meta = DPPAssessmentMetadata(
            transcription_confidence=0.65,
            acoustic_features_available=False,
            analysis_duration_seconds=3.2,
            quality_warning="Low audio quality",
        )
        result = meta.to_dict()

        assert result["quality_warning"] == "Low audio quality"

    def test_dpp_risk_assessment_to_dict(self):
        from src.text_processing.gpt4_synthesizer import (
            AnaliseDPP,
            ComponenteAnalise,
            DPPAssessmentMetadata,
            DPPRiskAssessment,
        )

        assessment = DPPRiskAssessment(
            consultation_id="test-123",
            analysis_timestamp="2026-02-14T10:00:00Z",
            analise_dpp=AnaliseDPP(
                probabilidade_percentual=30,
                nivel_risco="Moderado",
                indicadores_detectados=["Fadiga"],
                sugerir_alerta=False,
                justificativa_clinica="Alguns indicadores leves.",
                confianca_analise=0.8,
                componentes_analise=ComponenteAnalise(0.7, 0.3),
            ),
            metadata=DPPAssessmentMetadata(
                transcription_confidence=0.9,
                acoustic_features_available=True,
                analysis_duration_seconds=4.0,
            ),
        )
        result = assessment.to_dict()

        assert result["consultation_id"] == "test-123"
        assert result["analysis_timestamp"] == "2026-02-14T10:00:00Z"
        assert "analise_dpp" in result
        assert "metadata" in result


class TestFormatFunctions:
    """Testes para funções de formatação."""

    def test_format_acoustic_summary_none(self):
        from src.text_processing.gpt4_synthesizer import _format_acoustic_summary

        result = _format_acoustic_summary(None)
        assert result == "Dados acústicos não disponíveis."

    def test_format_acoustic_summary_empty(self):
        from src.text_processing.gpt4_synthesizer import _format_acoustic_summary

        # Empty dict is treated same as None (no acoustic data)
        result = _format_acoustic_summary({})
        assert "Dados" in result
        assert len(result) > 10

    def test_format_acoustic_summary_with_hesitations(self):
        from src.text_processing.gpt4_synthesizer import _format_acoustic_summary

        data = {
            "hesitation_markers": [
                {"type": "pause", "duration": 2.0},
                {"type": "pause", "duration": 1.5},
                {"type": "filler_word", "text": "então"},
            ],
            "confidence": 0.85,
            "duration": 60.0,
        }
        result = _format_acoustic_summary(data)

        assert "Pausas longas: 2" in result
        assert "Filler words: 1" in result
        assert "85%" in result
        assert "60.0s" in result

    def test_format_user_message(self):
        from src.text_processing.gpt4_synthesizer import _format_user_message

        result = _format_user_message(
            "Eu estou me sentindo muito cansada.",
            {"confidence": 0.9, "duration": 30.0},
        )

        assert "Eu estou me sentindo muito cansada" in result
        assert "Transcrição" in result
        assert "Acústica" in result


class TestParseGptResponse:
    """Testes para parsing da resposta GPT."""

    def test_parse_valid_json(self):
        from src.text_processing.gpt4_synthesizer import _parse_gpt_response

        response = json.dumps(
            {
                "probabilidade_percentual": 45,
                "nivel_risco": "Moderado",
                "indicadores_detectados": ["Fadiga"],
                "justificativa_clinica": "Indicadores moderados.",
                "confianca_analise": 0.8,
            }
        )
        result = _parse_gpt_response(response)

        assert result["probabilidade_percentual"] == 45
        assert result["nivel_risco"] == "Moderado"

    def test_parse_json_with_markdown(self):
        from src.text_processing.gpt4_synthesizer import _parse_gpt_response

        response = """```json
{
    "probabilidade_percentual": 60,
    "nivel_risco": "Alto",
    "indicadores_detectados": ["Desamparo"],
    "justificativa_clinica": "Indicadores altos.",
    "confianca_analise": 0.75
}
```"""
        result = _parse_gpt_response(response)

        assert result["probabilidade_percentual"] == 60
        assert result["nivel_risco"] == "Alto"

    def test_parse_invalid_json(self):
        from src.text_processing.gpt4_synthesizer import _parse_gpt_response

        with pytest.raises(ValueError) as exc_info:
            _parse_gpt_response("not valid json")

        assert "JSON válido" in str(exc_info.value)

    def test_parse_missing_field(self):
        from src.text_processing.gpt4_synthesizer import _parse_gpt_response

        response = json.dumps(
            {
                "probabilidade_percentual": 50,
                # Missing other required fields
            }
        )
        with pytest.raises(ValueError) as exc_info:
            _parse_gpt_response(response)

        assert "Campo obrigatório ausente" in str(exc_info.value)

    def test_parse_invalid_probability_range(self):
        from src.text_processing.gpt4_synthesizer import _parse_gpt_response

        response = json.dumps(
            {
                "probabilidade_percentual": 150,  # Invalid
                "nivel_risco": "Alto",
                "indicadores_detectados": [],
                "justificativa_clinica": "Test",
                "confianca_analise": 0.5,
            }
        )
        with pytest.raises(ValueError) as exc_info:
            _parse_gpt_response(response)

        assert "0-100" in str(exc_info.value)

    def test_parse_invalid_risk_level(self):
        from src.text_processing.gpt4_synthesizer import _parse_gpt_response

        response = json.dumps(
            {
                "probabilidade_percentual": 50,
                "nivel_risco": "InvalidLevel",
                "indicadores_detectados": [],
                "justificativa_clinica": "Test",
                "confianca_analise": 0.5,
            }
        )
        with pytest.raises(ValueError) as exc_info:
            _parse_gpt_response(response)

        assert "nivel_risco inválido" in str(exc_info.value)

    def test_parse_default_weights(self):
        from src.text_processing.gpt4_synthesizer import _parse_gpt_response

        response = json.dumps(
            {
                "probabilidade_percentual": 50,
                "nivel_risco": "Moderado",
                "indicadores_detectados": [],
                "justificativa_clinica": "Test",
                "confianca_analise": 0.5,
                # No peso_textual or peso_acustico
            }
        )
        result = _parse_gpt_response(response)

        assert result["peso_textual"] == 0.7
        assert result["peso_acustico"] == 0.3


class TestSynthesizeValidation:
    """Testes de validação para synthesize_dpp_analysis."""

    def test_empty_transcription_raises(self):
        from src.text_processing.gpt4_synthesizer import synthesize_dpp_analysis

        with pytest.raises(ValueError) as exc_info:
            synthesize_dpp_analysis("")

        assert "vazia" in str(exc_info.value)

    def test_whitespace_transcription_raises(self):
        from src.text_processing.gpt4_synthesizer import synthesize_dpp_analysis

        with pytest.raises(ValueError) as exc_info:
            synthesize_dpp_analysis("   ")

        assert "vazia" in str(exc_info.value)


class TestSynthesizeMocked:
    """Testes com mock da API OpenAI."""

    @pytest.fixture
    def mock_gpt_response(self):
        """Cria resposta mock do GPT-4."""

        class MockChoice:
            def __init__(self):
                self.message = mock.MagicMock()
                self.message.content = json.dumps(
                    {
                        "probabilidade_percentual": 72,
                        "nivel_risco": "Alto",
                        "indicadores_detectados": [
                            "Fadiga extrema",
                            "Sentimento de incapacidade",
                        ],
                        "justificativa_clinica": (
                            "Paciente apresenta múltiplos indicadores de DPP."
                        ),
                        "confianca_analise": 0.85,
                        "peso_textual": 0.65,
                        "peso_acustico": 0.35,
                    }
                )

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        return MockResponse()

    def test_synthesize_success(self, mock_gpt_response):
        from src.text_processing.gpt4_synthesizer import synthesize_dpp_analysis

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock_gpt_response

        mock_openai_module = mock.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with mock.patch.dict("sys.modules", {"openai": mock_openai_module}):
                result = synthesize_dpp_analysis(
                    "Eu estou muito cansada e não consigo cuidar do bebê.",
                    acoustic_data={"confidence": 0.9, "duration": 60.0},
                    consultation_id="test-001",
                )

        assert result.consultation_id == "test-001"
        assert result.analise_dpp.probabilidade_percentual == 72
        assert result.analise_dpp.nivel_risco == "Alto"
        assert result.analise_dpp.sugerir_alerta is False  # 72 < 75
        assert len(result.analise_dpp.indicadores_detectados) == 2
        assert result.metadata.acoustic_features_available is True

    def test_synthesize_triggers_alert_at_75(self):
        from src.text_processing.gpt4_synthesizer import synthesize_dpp_analysis

        mock_response = mock.MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "probabilidade_percentual": 80,
                "nivel_risco": "Crítico",
                "indicadores_detectados": ["Indicador crítico"],
                "justificativa_clinica": "Risco elevado.",
                "confianca_analise": 0.9,
            }
        )

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_module = mock.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with mock.patch.dict("sys.modules", {"openai": mock_openai_module}):
                result = synthesize_dpp_analysis("Texto de teste.")

        assert result.analise_dpp.probabilidade_percentual == 80
        assert result.analise_dpp.sugerir_alerta is True

    def test_synthesize_retry_on_invalid_json(self):
        from src.text_processing.gpt4_synthesizer import synthesize_dpp_analysis

        # First response is invalid, second is valid
        invalid_response = mock.MagicMock()
        invalid_response.choices[0].message.content = "not valid json"

        valid_response = mock.MagicMock()
        valid_response.choices[0].message.content = json.dumps(
            {
                "probabilidade_percentual": 30,
                "nivel_risco": "Moderado",
                "indicadores_detectados": [],
                "justificativa_clinica": "Baixo risco.",
                "confianca_analise": 0.7,
            }
        )

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = [
            invalid_response,
            valid_response,
        ]

        mock_openai_module = mock.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with mock.patch.dict("sys.modules", {"openai": mock_openai_module}):
                result = synthesize_dpp_analysis("Texto de teste.")

        assert result.analise_dpp.probabilidade_percentual == 30
        assert mock_client.chat.completions.create.call_count == 2

    def test_synthesize_all_retries_fail(self):
        from src.text_processing.gpt4_synthesizer import synthesize_dpp_analysis

        mock_response = mock.MagicMock()
        mock_response.choices[0].message.content = "invalid json"

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_module = mock.MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with mock.patch.dict("sys.modules", {"openai": mock_openai_module}):
                with pytest.raises(RuntimeError) as exc_info:
                    synthesize_dpp_analysis("Texto de teste.")

        assert "falhou após 3 tentativas" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 3


class TestFallbackAssessment:
    """Testes para create_fallback_assessment."""

    def test_fallback_creates_conservative_assessment(self):
        from src.text_processing.gpt4_synthesizer import create_fallback_assessment

        result = create_fallback_assessment(
            consultation_id="fallback-001",
            transcription="Texto original.",
            error_message="API timeout",
        )

        assert result.consultation_id == "fallback-001"
        assert result.analise_dpp.probabilidade_percentual == 0
        assert result.analise_dpp.nivel_risco == "Baixo"
        assert result.analise_dpp.sugerir_alerta is False
        assert result.analise_dpp.confianca_analise == 0.0
        assert "Fallback" in result.metadata.quality_warning
        assert "revisão manual" in result.analise_dpp.justificativa_clinica


class TestRiskLevel:
    """Testes para enum RiskLevel."""

    def test_risk_level_values(self):
        from src.text_processing.gpt4_synthesizer import RiskLevel

        assert RiskLevel.BAIXO.value == "Baixo"
        assert RiskLevel.MODERADO.value == "Moderado"
        assert RiskLevel.ALTO.value == "Alto"
        assert RiskLevel.CRITICO.value == "Crítico"
