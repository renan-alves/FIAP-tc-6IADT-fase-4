"""Pacote de processamento de texto."""

from . import text_analyzer
from .gpt4_synthesizer import (
    AnaliseDPP,
    ComponenteAnalise,
    DPPAssessmentMetadata,
    DPPRiskAssessment,
    RiskLevel,
    create_fallback_assessment,
    synthesize_dpp_analysis,
)

__all__ = [
    "text_analyzer",
    "synthesize_dpp_analysis",
    "create_fallback_assessment",
    "DPPRiskAssessment",
    "AnaliseDPP",
    "ComponenteAnalise",
    "DPPAssessmentMetadata",
    "RiskLevel",
]
