from ultralytics import YOLO
from pathlib import Path

def download_dataset():
    model = YOLO("yolov8n.pt")
    return "coco8.yaml"

if __name__ == "__main__":
    download_dataset()

