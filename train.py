from ultralytics import YOLO 

model = YOLO('yolov8n.pt') 

results = model.train( 
    data='/Users/mohitkumar/surakshit-sadak-ai/helmet_dataset/data.yaml', 
    epochs=10, 
    project='runs', 
    name='surakshit_sadak_local',
    device='mps'  # <-- THIS IS THE MAGIC WORD FOR MAC M-CHIPS
)
