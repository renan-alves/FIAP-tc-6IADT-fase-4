"""Script de teste para pipeline completo: transcrição + síntese DPP."""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio_processing.whisper_client import transcribe_audio
from src.text_processing.gpt4_synthesizer import synthesize_dpp_analysis


def test_full_pipeline(audio_path: Path):
    """Testa pipeline completo: transcrição + síntese DPP."""
    print(f"\n{'=' * 60}")
    print(f"Analisando: {audio_path}")
    print("=" * 60)

    # Fase 1: Transcrição
    print("\n🎤 Transcrevendo áudio...")
    transcription = transcribe_audio(audio_path)

    print(f"\n📝 Texto ({len(transcription.text)} caracteres):")
    print("-" * 40)
    text_preview = transcription.text[:500]
    if len(transcription.text) > 500:
        text_preview += "..."
    print(text_preview)

    print(f"\n📊 Estatísticas da transcrição:")
    print(f"  - Segmentos: {len(transcription.segments)}")
    print(f"  - Marcadores de hesitação: {len(transcription.hesitation_markers)}")
    print(f"  - Confiança: {transcription.confidence:.2%}")
    print(f"  - Duração: {transcription.duration:.1f}s")

    # Preparar dados acústicos para síntese
    acoustic_data = {
        "hesitation_markers": [m.to_dict() for m in transcription.hesitation_markers],
        "confidence": transcription.confidence,
        "duration": transcription.duration,
    }

    # Fase 2: Síntese DPP via GPT-4
    print("\n🧠 Sintetizando análise de risco DPP...")
    assessment = synthesize_dpp_analysis(
        transcription=transcription.text,
        acoustic_data=acoustic_data,
        consultation_id=audio_path.stem,
    )

    print(f"\n🎯 Avaliação de Risco DPP:")
    print("-" * 40)
    dpp = assessment.analise_dpp
    print(f"  Probabilidade: {dpp.probabilidade_percentual}%")
    print(f"  Nível de Risco: {dpp.nivel_risco}")
    alert_status = "SIM ⚠️" if dpp.sugerir_alerta else "Não"
    print(f"  Sugerir Alerta: {alert_status}")
    print(f"  Confiança: {dpp.confianca_analise:.2%}")

    if dpp.indicadores_detectados:
        print(f"\n📋 Indicadores Detectados:")
        for ind in dpp.indicadores_detectados:
            print(f"  • {ind}")

    print(f"\n💬 Justificativa Clínica:")
    print(f"  {dpp.justificativa_clinica}")

    print(f"\n⚖️ Pesos da Análise:")
    comp = dpp.componentes_analise
    print(f"  - Componente Textual: {comp.componente_textual_peso:.0%}")
    print(f"  - Componente Acústico: {comp.componente_acustico_peso:.0%}")

    return assessment


if __name__ == "__main__":
    example_dir = Path("docs/example")

    results = []
    for audio_file in sorted(example_dir.glob("*.mp4")):
        try:
            assessment = test_full_pipeline(audio_file)
            results.append(assessment.to_dict())
        except Exception as e:
            print(f"\n❌ Erro ao processar {audio_file}: {e}")

    # Salvar resultados
    if results:
        output_path = example_dir / "dpp_analysis_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Resultados salvos em: {output_path}")
    else:
        print("\n⚠️ Nenhum resultado para salvar.")
