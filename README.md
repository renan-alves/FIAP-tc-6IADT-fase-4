<div align="center">

# Análise de Vídeos Cirúrgicos + Detecção de Risco DPP

**Sistema multimodal de IA para detecção de instrumentos cirúrgicos e avaliação de risco de depressão pós-parto**

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://ultralytics.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper%20%2B%20GPT--4-412991?style=flat-square&logo=openai)](https://openai.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

[Visão Geral](#visão-geral) • [Funcionalidades](#funcionalidades) • [Primeiros Passos](#primeiros-passos) • [Uso](#uso) • [Testes](#testes) • [Referência da API](#referência-da-api)

</div>

## Visão Geral

Este projeto combina visão computacional e processamento de linguagem natural para aplicações em saúde:

- **Fase 1**: Detecção de instrumentos cirúrgicos em vídeo usando YOLOv8
- **Fase 2**: Transcrição de áudio e síntese clínica de risco para triagem de Depressão Pós-Parto (DPP)

O sistema oferece interfaces CLI e REST API para integração em fluxos de trabalho clínicos.

> [!NOTE]
> Este é um projeto acadêmico. Não substitui avaliação médica profissional.

## Funcionalidades

- **Análise de Vídeo**: Detecção de instrumentos cirúrgicos baseada em YOLOv8 com análise quadro a quadro
- **Transcrição de Áudio**: Transcrição em português (pt-BR) via OpenAI Whisper com marcadores de hesitação
- **Síntese Clínica**: Avaliação de risco DPP via GPT-4 com scores de probabilidade e justificativa clínica
- **Interfaces Duplas**: CLI para processamento local/batch, REST API para gerenciamento assíncrono de jobs
- **Saída Estruturada**: Probabilidade de risco (0-100%), nível de risco, indicadores detectados e fundamentação clínica

## Primeiros Passos

### Pré-requisitos

- Python 3.10+
- Chave de API OpenAI (para funcionalidades da Fase 2)

### Instalação

```bash
git clone <repo>
cd FIAP-tc-6IADT-fase-4
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Configuração

```bash
# Obrigatório para análise de áudio/texto
export OPENAI_API_KEY=sk-sua-chave-aqui

# Opcional
export API_BEARER_TOKEN=seu-token-secreto  # Autenticação da REST API
export LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR
```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY = "sk-sua-chave-aqui"
$env:API_BEARER_TOKEN = "seu-token-secreto"
```

## Uso

### Análise de Vídeo (Fase 1)

```python
from src.detection.yolo_detector import YOLODetector
from src.video_processing.video_analyzer import VideoAnalyzer

detector = YOLODetector()
analyzer = VideoAnalyzer(detector=detector)
results = analyzer.analyze_video("video_cirurgia.mp4")
```

### Treinamento do Modelo

```bash
# Treinamento local (CPU)
python train_model.py

# Google Colab (GPU recomendado)
# Faça upload de train_colab.ipynb e selecione runtime T4
```

### CLI - Análise de Áudio (Fase 2)

```bash
python -m src.cli analyze <arquivo_audio> [opções]
```

| Opção | Descrição |
|-------|-----------|
| `--output <caminho>` | Salvar resultado em arquivo (padrão: stdout) |
| `--format <json\|text>` | Formato de saída (padrão: json) |
| `--consultation-id <id>` | ID da consulta (gerado automaticamente se omitido) |
| `--verbose` | Habilitar logging detalhado |
| `--no-cleanup` | Preservar arquivos temporários |

**Exemplos:**

```bash
# Análise básica (JSON para stdout)
python -m src.cli analyze consulta.wav

# Salvar em arquivo
python -m src.cli analyze consulta.wav --output resultado.json

# Saída legível por humanos
python -m src.cli analyze consulta.wav --format text

# Modo debug
python -m src.cli analyze consulta.wav --verbose
```

**Códigos de Saída:**

| Código | Significado |
|--------|-------------|
| 0 | Sucesso |
| 1 | Arquivo inválido ou argumentos incorretos |
| 2 | Erro no processamento de áudio |
| 3 | Erro na API OpenAI |
| 4 | Erro de configuração (chave ausente) |

**Processamento em Lote (Linux/macOS):**

```bash
# Processar múltiplos arquivos em paralelo
ls data/*.wav | parallel -j 5 'python -m src.cli analyze {} --output results/{/.}.json'
```

## Referência da API

### Iniciar Servidor

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/analyze` | Submete áudio para análise (retorna job_id) |
| `GET` | `/analyze/{job_id}` | Consulta status/resultado do job |
| `GET` | `/health` | Health check com status das dependências |
| `GET` | `/docs` | Documentação Swagger UI |

### Submeter Análise

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Authorization: Bearer $API_BEARER_TOKEN" \
  -F "audio_file=@consulta.wav" \
  -F "consultation_id=consul-001"
```

**Resposta (HTTP 202):**

```json
{
  "job_id": "abc123-uuid",
  "correlation_id": "xyz789-uuid"
}
```

### Consultar Resultado

```bash
curl http://localhost:8000/analyze/abc123-uuid \
  -H "Authorization: Bearer $API_BEARER_TOKEN"
```

**Resposta (HTTP 200 - Concluído):**

```json
{
  "job_id": "abc123-uuid",
  "status": "completed",
  "result": {
    "analise_dpp": {
      "probabilidade_percentual": 72,
      "nivel_risco": "Alto",
      "sugerir_alerta": false,
      "indicadores_detectados": ["fadiga verbalizada", "tom depressivo"],
      "justificativa_clinica": "..."
    }
  }
}
```

**Formatos de áudio suportados:** `.wav`, `.mp3`, `.mp4`, `.m4a`, `.flac`, `.ogg`, `.webm`

**Limite de tamanho:** 30 MB

## Testes

### Testes Unitários

```bash
# Executar todos os testes
pytest

# Com relatório de cobertura
pytest --cov=src --cov-report=term-missing

# Módulos de teste específicos
pytest tests/test_api.py -v      # Testes da REST API
pytest tests/test_cli.py -v      # Testes do CLI
pytest tests/test_detection.py -v # Testes de detecção YOLOv8
```

### Testes de Integração (sem OpenAI)

```bash
pytest tests/ --ignore=tests/test_orchestrator.py --ignore=tests/test_whisper_client.py --ignore=tests/test_gpt4_synthesizer.py
```

### Testes que Requerem OpenAI

```bash
# Requer OPENAI_API_KEY configurada
pytest tests/test_whisper_client.py tests/test_gpt4_synthesizer.py tests/test_orchestrator.py -v
```

**Suíte de testes atual:** 60 testes (excluindo testes dependentes de OpenAI)

### Scripts de Teste do Pipeline

O diretório `scripts/` contém scripts de teste para validação do pipeline completo de análise de áudio com arquivos reais.

#### Teste de Transcrição Whisper

Testa a transcrição de áudio usando OpenAI Whisper:

```bash
python scripts/test_whisper_example.py
```

Este script:
- Processa todos os arquivos `.mp4` em `docs/example/`
- Exibe texto da transcrição, contagem de segmentos e confiança
- Mostra marcadores de hesitação detectados (pausas e fillers)

**Exemplo de saída:**

```
============================================================
Transcrevendo: docs/example/sample.mp4
============================================================

📝 Texto (1234 caracteres):
----------------------------------------
[texto da transcrição...]

📊 Estatísticas:
  - Segmentos: 15
  - Marcadores de hesitação: 3
  - Confiança: 95.2%
  - Duração: 45.3s
  - Idioma: pt

🔍 Marcadores de hesitação:
  - PAUSA: 2.1s em 12.5s
  - FILLER: 'éh' em 23.1s
```

#### Teste do Pipeline Completo

Testa o pipeline completo (transcrição + síntese DPP):

```bash
python scripts/test_full_pipeline.py
```

Este script:
- Transcreve arquivos de áudio de `docs/example/`
- Envia transcrição para GPT-4 para síntese clínica de risco
- Exibe avaliação de risco DPP com probabilidade, indicadores e justificativa
- Salva resultados em `docs/example/dpp_analysis_results.json`

**Exemplo de saída:**

```
============================================================
Analisando: docs/example/consulta.mp4
============================================================

🎤 Transcrevendo áudio...

📝 Texto (500 caracteres):
----------------------------------------
[prévia da transcrição...]

📊 Estatísticas da transcrição:
  - Segmentos: 12
  - Marcadores de hesitação: 4
  - Confiança: 92.5%
  - Duração: 38.2s

🧠 Sintetizando análise de risco DPP...

🎯 Avaliação de Risco DPP:
----------------------------------------
  Probabilidade: 65%
  Nível de Risco: Moderado
  Sugerir Alerta: Não
  Confiança: 88.0%

📋 Indicadores Detectados:
  • fadiga verbalizada
  • alteração no padrão de sono

💬 Justificativa Clínica:
  A paciente apresenta sinais moderados...

⚖️ Pesos da Análise:
  - Componente Textual: 70%
  - Componente Acústico: 30%
```

> [!IMPORTANT]
> Ambos os scripts requerem `OPENAI_API_KEY` configurada e arquivos de áudio de exemplo em `docs/example/`.

## Estrutura do Projeto

```
src/
├── api/                 # REST API (FastAPI)
├── cli/                 # Interface de linha de comando
├── detection/           # Detector YOLOv8
├── video_processing/    # Analisador de vídeo
├── audio_processing/    # Cliente de transcrição Whisper
├── text_processing/     # Síntese clínica GPT-4
├── integration/         # Orquestrador, cache de jobs, logs de auditoria
└── reports/             # Gerador de relatórios

scripts/                 # Scripts de teste do pipeline
models/                  # Modelos YOLO treinados
data/                    # Arquivos de dados e relatórios
config/                  # Configurações e logging
tests/                   # Suíte de testes pytest
specs/                   # Especificações técnicas
docs/example/            # Arquivos de áudio/vídeo de exemplo
```

## Segurança

- Nunca armazene chaves de API no código; use variáveis de ambiente
- Arquivos temporários são limpos automaticamente após processamento
- Para produção, habilite TLS e revise requisitos de conformidade LGPD
- Logs de auditoria armazenados em `data/audit_logs/` (sem PII)

## Recursos

- [Documentação YOLOv8](https://docs.ultralytics.com)
- [API OpenAI Whisper](https://platform.openai.com/docs/guides/speech-to-text)
- [API OpenAI GPT-4](https://platform.openai.com/docs/guides/text-generation)
- [Documentação FastAPI](https://fastapi.tiangolo.com)

