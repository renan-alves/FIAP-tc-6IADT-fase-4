"""Script de teste para transcrição Whisper com arquivos de exemplo."""

from pathlib import Path
import json

from src.audio_processing.whisper_client import transcribe_audio


def test_transcription(audio_path: Path):
    """Testa transcrição de um arquivo de áudio."""
    print(f"\n{'=' * 60}")
    print(f"Transcrevendo: {audio_path}")
    print("=" * 60)

    result = transcribe_audio(audio_path)

    print(f"\n📝 Texto ({len(result.text)} caracteres):")
    print("-" * 40)
    print(result.text)

    print(f"\n📊 Estatísticas:")
    print(f"  - Segmentos: {len(result.segments)}")
    print(f"  - Marcadores de hesitação: {len(result.hesitation_markers)}")
    print(f"  - Confiança: {result.confidence:.2%}")
    print(f"  - Duração: {result.duration:.1f}s")
    print(f"  - Idioma: {result.language}")

    if result.hesitation_markers:
        print(f"\n🔍 Marcadores de hesitação:")
        for m in result.hesitation_markers:
            if m.type == "pause":
                print(f"  - PAUSA: {m.duration:.1f}s em {m.start_time:.1f}s")
            else:
                print(f"  - FILLER: '{m.text}' em {m.start_time:.1f}s")

    return result


if __name__ == "__main__":
    example_dir = Path("docs/example")

    for audio_file in sorted(example_dir.glob("*.mp4")):
        try:
            test_transcription(audio_file)
        except Exception as e:
            print(f"\n❌ Erro ao processar {audio_file}: {e}")

    print("\n✅ Testes concluídos!")
