from ultralytics import YOLO
import os

MODEL_NAME = 'yolov8n.pt' 

DATA_CONFIG = 'master_data.yaml'

EPOCHS = 100            
IMG_SIZE = 640          
BATCH_SIZE = 16         
PROJECT_NAME = 'Airport_Safety_FYP'
RUN_NAME = 'final_run_v1' 

def main():
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print("Starting Training...")
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        
        patience=20,      
        save=True,        
        device=0,        
        workers=4,       
        exist_ok=True,    
        pretrained=True,  
        optimizer='auto', 
        verbose=True      
    )

    print(f"Training Complete! Model is saved in: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")

if __name__ == '__main__':
    main()