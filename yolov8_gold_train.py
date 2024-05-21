from ultralytics import YOLO

# Load a model
model = YOLO('yolov8-gold.yaml')  # build a new model from YAML

# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco-pose.yaml', optimizer='AdamW',epochs=5, imgsz=640,lr0=0.01)
# results = model.predict('/root/ultralytics/ultralytics/assets/bus.jpg',save=True)