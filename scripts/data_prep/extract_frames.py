import cv2
import os
from pathlib import Path

VIDEO_INPUT_DIR = Path("data/raw/airport_videos") 

OUTPUT_DIR = Path("data/raw/airport_scraped")

SECONDS_INTERVAL = 1 

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
video_files = [p for p in VIDEO_INPUT_DIR.iterdir() if p.suffix.lower() in video_extensions]

print(f"🚀 Found {len(video_files)} videos. Starting extraction...")

total_images = 0

for video_path in video_files:
    print(f"[*] Processing: {video_path.name}...")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 
    
    frame_interval = int(fps * SECONDS_INTERVAL)
    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % frame_interval == 0:
            safe_name = video_path.stem.replace(" ", "_")
            out_name = f"{safe_name}_{saved_count}.jpg"
            
            cv2.imwrite(str(OUTPUT_DIR / out_name), frame)
            saved_count += 1
            total_images += 1
            
        count += 1
        
    cap.release()

print(f"✅ DONE! Extracted {total_images} images to {OUTPUT_DIR}")
print(f"⚠️ NEXT STEP: Run 'batch_anonymize.py' on this folder BEFORE labeling!")