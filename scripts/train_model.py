from ultralytics import YOLO
from pathlib import Path

DATA_YAML = "coco8.yaml"  # Dataset de exemplo - será baixado automaticamente
# DATA_YAML = r"C:\caminho\para\seu\data.yaml"  # Descomente e ajuste

MODEL = "yolov8s.pt"
EPOCHS = 50  # Reduzido para teste rápido
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cpu"
PATIENCE = 20

def train_model():
    model = YOLO(MODEL)
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            patience=PATIENCE,
            save=True,
            project='runs/detect',
            name='surgical_instruments',
            verbose=True
        )
        print("\n✅ Training completed!")
        
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    train_model()
