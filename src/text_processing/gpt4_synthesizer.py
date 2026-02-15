"""Síntese clínica de risco de DPP via GPT-4.

Este módulo integra transcrição e metadados acústicos para produzir uma
avaliação estruturada de risco de Depressão Pós-Parto (DPP) usando GPT-4.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from config.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Configuração de retry
MAX_RETRIES = 3
BACKOFF_BASE_MS = 200

# Threshold para sugestão de alerta
ALERT_THRESHOLD_PERCENT = 75


class RiskLevel(str, Enum):
    """Níveis de risco para DPP."""

    BAIXO = "Baixo"
    MODERADO = "Moderado"
    ALTO = "Alto"
    CRITICO = "Crítico"


@dataclass
class ComponenteAnalise:
    """Pesos dos componentes da análise."""

    componente_textual_peso: float
    componente_acustico_peso: float

    def to_dict(self) -> dict:
        return {
            "componente_textual_peso": self.componente_textual_peso,
            "componente_acustico_peso": self.componente_acustico_peso,
        }


@dataclass
class AnaliseDPP:
    """Resultado da análise de risco de DPP."""

    probabilidade_percentual: int  # 0-100
    nivel_risco: str  # Baixo, Moderado, Alto, Crítico
    indicadores_detectados: List[str]
    sugerir_alerta: bool
    justificativa_clinica: str
    confianca_analise: float  # 0.0-1.0
    componentes_analise: ComponenteAnalise

    def to_dict(self) -> dict:
        return {
            "probabilidade_percentual": self.probabilidade_percentual,
            "nivel_risco": self.nivel_risco,
            "indicadores_detectados": self.indicadores_detectados,
            "sugerir_alerta": self.sugerir_alerta,
            "justificativa_clinica": self.justificativa_clinica,
            "confianca_analise": self.confianca_analise,
            "componentes_analise": self.componentes_analise.to_dict(),
        }


@dataclass
class DPPAssessmentMetadata:
    """Metadados da avaliação de DPP."""

    transcription_confidence: float
    acoustic_features_available: bool
    analysis_duration_seconds: float
    quality_warning: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "transcription_confidence": self.transcription_confidence,
            "acoustic_features_available": self.acoustic_features_available,
            "analysis_duration_seconds": self.analysis_duration_seconds,
        }
        if self.quality_warning:
            result["quality_warning"] = self.quality_warning
        return result


@dataclass
class DPPRiskAssessment:
    """Avaliação completa de risco de DPP."""

    consultation_id: str
    analysis_timestamp: str  # ISO-8601
    analise_dpp: AnaliseDPP
    metadata: DPPAssessmentMetadata

    def to_dict(self) -> dict:
        return {
            "consultation_id": self.consultation_id,
            "analysis_timestamp": self.analysis_timestamp,
            "analise_dpp": self.analise_dpp.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


# System prompt baseado na spec para análise clínica de DPP
SYSTEM_PROMPT = """Você é um assistente clínico especializado em saúde mental materna, com foco em detecção de Depressão Pós-Parto (DPP). Sua função é analisar transcrições de consultas clínicas e metadados acústicos para identificar indicadores de DPP.

## Contexto Clínico
A Depressão Pós-Parto afeta aproximadamente 10-20% das mulheres no período pós-parto. Os principais indicadores incluem:
- Sentimentos de incapacidade materna ou culpa excessiva
- Fadiga extrema além do esperado
- Dificuldade de conexão/vínculo com o bebê
- Isolamento social
- Alterações de apetite ou sono significativas
- Pensamentos negativos recorrentes
- Choro frequente ou irritabilidade extrema
- Desinteresse em atividades antes prazerosas
- Verbalização de desamparo ou desesperança

## Marcadores Acústicos Relevantes
- Pitch baixo (frequência fundamental reduzida)
- Jitter/shimmer elevados (instabilidade vocal)
- Pausas longas e frequentes (hesitação)
- Ritmo de fala lento
- Filler words excessivos

## Formato de Resposta
Você DEVE responder APENAS com um objeto JSON válido no seguinte formato, sem texto adicional:

{
  "probabilidade_percentual": <número inteiro 0-100>,
  "nivel_risco": "<Baixo|Moderado|Alto|Crítico>",
  "indicadores_detectados": ["<indicador 1>", "<indicador 2>", ...],
  "justificativa_clinica": "<explicação em português do Brasil>",
  "confianca_analise": <número 0.0-1.0>,
  "peso_textual": <número 0.0-1.0>,
  "peso_acustico": <número 0.0-1.0>
}

## Regras de Classificação
- Baixo (0-24%): Sem indicadores significativos
- Moderado (25-49%): Alguns indicadores presentes, monitoramento recomendado
- Alto (50-74%): Múltiplos indicadores, avaliação clínica recomendada
- Crítico (75-100%): Indicadores severos, intervenção urgente necessária

## Regras de Pesos
- Se metadados acústicos estiverem disponíveis: peso_textual ~0.6, peso_acustico ~0.4
- Se apenas texto disponível: peso_textual = 1.0, peso_acustico = 0.0
- Ajuste os pesos baseado na qualidade e relevância de cada componente"""


def _get_openai_client():
    """Obtém cliente OpenAI configurado com API key do ambiente."""
    try:
        from openai import OpenAI
    except ImportError as e:
        logger.error("OpenAI SDK não instalado: %s", e)
        raise ImportError(
            "OpenAI SDK requerido. Instale com: pip install openai>=1.0"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY não configurada no ambiente")
        raise ValueError("OPENAI_API_KEY environment variable is required")

    return OpenAI(api_key=api_key)


def _format_acoustic_summary(acoustic_data: Optional[Dict[str, Any]]) -> str:
    """Formata resumo dos dados acústicos para o prompt."""
    if not acoustic_data:
        return "Dados acústicos não disponíveis."

    parts = []

    if "hesitation_markers" in acoustic_data:
        markers = acoustic_data["hesitation_markers"]
        pause_count = sum(1 for m in markers if m.get("type") == "pause")
        filler_count = sum(1 for m in markers if m.get("type") == "filler_word")
        if pause_count > 0:
            total_pause = sum(
                m.get("duration", 0) for m in markers if m.get("type") == "pause"
            )
            parts.append(f"Pausas longas: {pause_count} (total {total_pause:.1f}s)")
        if filler_count > 0:
            parts.append(f"Filler words: {filler_count}")

    if "confidence" in acoustic_data:
        parts.append(f"Confiança transcrição: {acoustic_data['confidence']:.0%}")

    if "duration" in acoustic_data:
        parts.append(f"Duração: {acoustic_data['duration']:.1f}s")

    return "; ".join(parts) if parts else "Análise acústica básica realizada."


def _format_user_message(
    transcription: str,
    acoustic_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Formata mensagem do usuário para análise GPT-4."""
    acoustic_summary = _format_acoustic_summary(acoustic_data)

    message = f"""Analise a seguinte transcrição de consulta clínica para identificar indicadores de Depressão Pós-Parto (DPP):

## Transcrição
{transcription}

## Análise Acústica
{acoustic_summary}

Forneça sua avaliação no formato JSON especificado."""

    return message


def _parse_gpt_response(response_text: str) -> Dict[str, Any]:
    """Extrai e valida JSON da resposta do GPT-4."""
    # Tentar extrair JSON do texto
    text = response_text.strip()

    # Remover markdown code blocks se presentes
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Falha ao parsear JSON da resposta GPT-4: %s", e)
        raise ValueError(f"Resposta GPT-4 não é JSON válido: {e}") from e

    # Validar campos obrigatórios
    required_fields = [
        "probabilidade_percentual",
        "nivel_risco",
        "indicadores_detectados",
        "justificativa_clinica",
        "confianca_analise",
    ]
    for field_name in required_fields:
        if field_name not in data:
            raise ValueError(f"Campo obrigatório ausente na resposta: {field_name}")

    # Validar tipos e ranges
    prob = data["probabilidade_percentual"]
    if not isinstance(prob, (int, float)) or prob < 0 or prob > 100:
        raise ValueError(f"probabilidade_percentual deve ser 0-100, recebido: {prob}")
    data["probabilidade_percentual"] = int(prob)

    nivel = data["nivel_risco"]
    valid_levels = {"Baixo", "Moderado", "Alto", "Crítico"}
    if nivel not in valid_levels:
        raise ValueError(f"nivel_risco inválido: {nivel}. Válidos: {valid_levels}")

    conf = data["confianca_analise"]
    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
        raise ValueError(f"confianca_analise deve ser 0.0-1.0, recebido: {conf}")
    data["confianca_analise"] = float(conf)

    # Defaults para pesos se não fornecidos
    data.setdefault("peso_textual", 0.7)
    data.setdefault("peso_acustico", 0.3)

    return data


def synthesize_dpp_analysis(
    transcription: str,
    acoustic_data: Optional[Dict[str, Any]] = None,
    consultation_id: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> DPPRiskAssessment:
    """Sintetiza análise de risco de DPP usando GPT-4.

    Args:
        transcription: Texto transcrito da consulta.
        acoustic_data: Dados acústicos opcionais (hesitation_markers, confidence, etc).
        consultation_id: ID da consulta (gerado se não fornecido).
        model: Modelo GPT a usar (default: gpt-4o-mini para custo-eficiência).

    Returns:
        DPPRiskAssessment com avaliação completa de risco.

    Raises:
        ValueError: Se OPENAI_API_KEY não configurada ou resposta inválida.
        RuntimeError: Se todas as tentativas falharem.
    """
    import uuid

    start_time = time.time()

    if not transcription or not transcription.strip():
        raise ValueError("Transcrição vazia não pode ser analisada")

    consultation_id = consultation_id or str(uuid.uuid4())
    client = _get_openai_client()

    user_message = _format_user_message(transcription, acoustic_data)
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                "Sintetizando análise DPP (tentativa %d/%d) com modelo %s",
                attempt + 1,
                MAX_RETRIES,
                model,
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,  # Baixa temperatura para consistência
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content
            logger.debug("Resposta GPT-4: %s", response_text[:500])

            # Parsear e validar resposta
            parsed = _parse_gpt_response(response_text)

            # Calcular duração
            duration = time.time() - start_time

            # Determinar se deve sugerir alerta
            sugerir_alerta = (
                parsed["probabilidade_percentual"] >= ALERT_THRESHOLD_PERCENT
            )

            # Construir resultado
            analise = AnaliseDPP(
                probabilidade_percentual=parsed["probabilidade_percentual"],
                nivel_risco=parsed["nivel_risco"],
                indicadores_detectados=parsed["indicadores_detectados"],
                sugerir_alerta=sugerir_alerta,
                justificativa_clinica=parsed["justificativa_clinica"],
                confianca_analise=parsed["confianca_analise"],
                componentes_analise=ComponenteAnalise(
                    componente_textual_peso=parsed["peso_textual"],
                    componente_acustico_peso=parsed["peso_acustico"],
                ),
            )

            # Extrair confiança da transcrição se disponível
            transcription_confidence = (
                acoustic_data.get("confidence", 0.0) if acoustic_data else 0.0
            )

            metadata = DPPAssessmentMetadata(
                transcription_confidence=transcription_confidence,
                acoustic_features_available=acoustic_data is not None,
                analysis_duration_seconds=round(duration, 2),
            )

            result = DPPRiskAssessment(
                consultation_id=consultation_id,
                analysis_timestamp=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                analise_dpp=analise,
                metadata=metadata,
            )

            logger.info(
                "Síntese DPP concluída: probabilidade=%d%%, risco=%s, alerta=%s",
                result.analise_dpp.probabilidade_percentual,
                result.analise_dpp.nivel_risco,
                result.analise_dpp.sugerir_alerta,
            )

            return result

        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(
                "Erro de validação (tentativa %d/%d): %s",
                attempt + 1,
                MAX_RETRIES,
                str(e),
            )
            # Retry imediatamente para erros de parsing
            continue

        except Exception as e:
            last_error = e
            logger.warning(
                "Erro na síntese DPP (tentativa %d/%d): %s",
                attempt + 1,
                MAX_RETRIES,
                str(e),
            )

            if attempt < MAX_RETRIES - 1:
                wait_ms = BACKOFF_BASE_MS * (2**attempt)
                logger.info("Aguardando %dms antes de retry...", wait_ms)
                time.sleep(wait_ms / 1000.0)

    # Todas as tentativas falharam
    error_msg = f"Síntese DPP falhou após {MAX_RETRIES} tentativas: {last_error}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_error


def create_fallback_assessment(
    consultation_id: str,
    transcription: str,
    error_message: str,
) -> DPPRiskAssessment:
    """Cria avaliação de fallback quando síntese GPT-4 falha.

    Usado para graceful degradation - permite que o pipeline continue
    mesmo se a síntese falhar, retornando uma avaliação conservadora.

    Args:
        consultation_id: ID da consulta.
        transcription: Texto transcrito (para referência).
        error_message: Mensagem de erro para incluir nos metadados.

    Returns:
        DPPRiskAssessment com avaliação conservadora.
    """
    logger.warning("Criando avaliação de fallback devido a: %s", error_message)

    analise = AnaliseDPP(
        probabilidade_percentual=0,
        nivel_risco="Baixo",
        indicadores_detectados=[],
        sugerir_alerta=False,
        justificativa_clinica=(
            "Avaliação automática indisponível. "
            "Recomenda-se revisão manual da transcrição."
        ),
        confianca_analise=0.0,
        componentes_analise=ComponenteAnalise(
            componente_textual_peso=0.0,
            componente_acustico_peso=0.0,
        ),
    )

    metadata = DPPAssessmentMetadata(
        transcription_confidence=0.0,
        acoustic_features_available=False,
        analysis_duration_seconds=0.0,
        quality_warning=f"Fallback: {error_message}",
    )

    return DPPRiskAssessment(
        consultation_id=consultation_id,
        analysis_timestamp=datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        analise_dpp=analise,
        metadata=metadata,
    )
