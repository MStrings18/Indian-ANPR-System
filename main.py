from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Using the nano model for quick test
results = model('https://ultralytics.com/images/bus.jpg', show=True)
