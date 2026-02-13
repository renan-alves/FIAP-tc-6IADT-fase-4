import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "data/reports"))
VIDEO_INPUT_DIR = Path(os.getenv("VIDEO_INPUT_DIR", "data/videos"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", ".temp"))

for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, VIDEO_INPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo_instruments/best.pt")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))

try:
    import torch
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu"

VIDEO_FRAME_RATE = int(os.getenv("VIDEO_FRAME_RATE", "30"))
VIDEO_RESIZE_WIDTH = int(os.getenv("VIDEO_RESIZE_WIDTH", "640"))
VIDEO_RESIZE_HEIGHT = int(os.getenv("VIDEO_RESIZE_HEIGHT", "480"))
VIDEO_SKIP_FRAMES = int(os.getenv("VIDEO_SKIP_FRAMES", "1"))

REPORT_FORMAT = os.getenv("REPORT_FORMAT", "pdf")
REPORT_LANGUAGE = os.getenv("REPORT_LANGUAGE", "pt-BR")

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/assistente.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))

OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "pt")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

MEDICAL_INSTRUMENTS = {
    "speculum": "Espéculo",
    "forceps": "Fórceps",
    "scissors": "Tesoura Cirúrgica",
    "curette": "Cureta",
    "dilator": "Dilatador",
    "probe": "Sonda",
    "clamp": "Pinça",
    "retractor": "Afastador",
    "trocar": "Trocarte",
    "light_source": "Fonte de Luz",
    "suction": "Aspirador",
}
