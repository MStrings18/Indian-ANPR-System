import torch
from ultralytics import YOLO


print(torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

print(torch.cuda.get_device_name(0))
print("Memory Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("Memory Cached:", torch.cuda.memory_reserved() / 1024**3, "GB")


# Create a new untrained model
model = YOLO("yolov8n.yaml")
results = model.train(data= 'config.yaml',
                      epochs = 300,
                      device='cuda',
                      half=True,
                      workers=4,
                     )