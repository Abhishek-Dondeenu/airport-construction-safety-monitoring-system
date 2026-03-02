from ultralytics import YOLO
import os
import glob
import random

MODEL_PATH = 'models/detect/Airport_Safety_FYP/final_run_v1/weights/best.pt'

SOURCE_FOLDER = 'data/raw/temp_marshalling/train/images'

OUTPUT_FOLDER = 'batch_inference_v1'

def main():
    print(f"🚀 Loading Model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"🔍 Scanning folder: {SOURCE_FOLDER}")
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in extensions:
        search_path = os.path.join(SOURCE_FOLDER, ext)
        found = glob.glob(search_path)
        image_files.extend(found)

    if not image_files:
        print("⚠️ No images found! Check your SOURCE_FOLDER path.")
        return

    random.shuffle(image_files)
    selected_images = image_files[:20]

    print(f"📸 Found {len(image_files)} images. Processing a random batch of {len(selected_images)}...")

    for i, img_path in enumerate(selected_images):
        filename = os.path.basename(img_path)
        print(f"   [{i+1}/{len(selected_images)}] Processing: {filename}")
        
        model.predict(
            source=img_path,
            save=True,
            conf=0.25,        
            iou=0.45,         
            project="runs/detect",
            name=OUTPUT_FOLDER,
            exist_ok=True,    
            verbose=False     
        )

    print(f"\n✅ Batch Test Complete!")
    print(f"📂 Open this folder to see the results:")
    print(f"   runs/detect/{OUTPUT_FOLDER}")

if __name__ == '__main__':
    main()