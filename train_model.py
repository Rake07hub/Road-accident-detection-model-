from ultralytics import YOLO
import torch

def main():

    print("starting training script")
    print("CUDA Available:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    print("loading model... OK")
    print("begin training...")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0   # IMPORTANT for Windows
    )

if __name__ == "__main__":
    main()