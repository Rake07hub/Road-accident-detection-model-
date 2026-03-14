from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train9/weights/best.pt")

cap = cv2.VideoCapture("testing_video\\videoplayback (online-video-cutter.com).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    frame = results[0].plot()

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()