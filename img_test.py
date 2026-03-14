from ultralytics import YOLO

model = YOLO("runs/detect/train9/weights/best.pt")

results = model("Accident_1730189984222_1730189988540.avif", show=True)