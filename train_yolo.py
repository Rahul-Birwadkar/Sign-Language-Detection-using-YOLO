from ultralytics import YOLO

# Load a YOLOv8 Nano model (small & fast)
model = YOLO('yolov8n.pt')  # Downloads if not available

# Train the model on your 5,000-image custom dataset
model.train(
    data=r'D:\Project\sign_language_yolo\Dataset\dataset\dataset.yaml',  # your dataset path
    epochs=10,          # reduced from 30 → 10 for faster results
    imgsz=416,          # smaller image size for faster training
    batch=8,            # smaller batch fits better on CPU/low memory
    project='sign_language_project',
    name='yolov8n_asl_5k',
    pretrained=True     # use pretrained weights
)
