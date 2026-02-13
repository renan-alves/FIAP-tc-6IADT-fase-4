# Detecção de Instrumentos Cirúrgicos com YOLOv8

Análise de vídeos cirúrgicos usando visão computacional para detecção de instrumentos.

## 🎯 Objetivo

Detecção automática de instrumentos em vídeos de procedimentos cirúrgicos.

## 🔬 Tecnologias

- YOLOv8 (Detecção de objetos)
- OpenCV (Processamento de vídeos)
- PyTorch
- Python 3.12

## 📁 Estrutura

```
src/
├── detection/           # YOLOv8 Detector
├── video_processing/    # Video analyzer
├── reports/             # Report generator
└── integration/         # Integration module

models/                  # Trained models
data/                    # Data and reports
config/                  # Settings
scripts/                 # Utility scripts
```

## 🚀 Instalação

```bash
git clone <repo>
cd FIAP-tc-6IADT-fase-4
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## 📖 Uso

```python
from src.detection.yolo_detector import YOLODetector
from src.video_processing.video_analyzer import VideoAnalyzer

detector = YOLODetector()
analyzer = VideoAnalyzer(detector=detector)
results = analyzer.analyze_video('video.mp4')
```

## 🏋️ Treinamento

**Local (CPU):**
```bash
python train_model.py
```

**Google Colab (GPU):**
1. Upload `train_colab.ipynb` to Colab
2. Select T4 GPU runtime
3. Run cells

## 🧪 Testes

```bash
pytest
```

## 📋 Status

- ✅ Phase 1: Video analysis with YOLOv8
- 📋 Phase 2: Audio analysis (ready)
- 📋 Phase 3: Text analysis (ready)

## ⚠️ Disclaimer

Academic project. Not a substitute for professional medical evaluation.
