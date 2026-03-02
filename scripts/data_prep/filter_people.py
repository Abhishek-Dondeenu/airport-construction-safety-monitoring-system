from ultralytics import YOLO
import os
import shutil
from pathlib import Path

INPUT_DIR = Path("data/processed/airport_scraped")
JUNK_DIR = Path("data/processed/airport_scraped/junk")
KEEP_DIR = Path("data/processed/airport_scraped/keep")

JUNK_DIR.mkdir(parents=True, exist_ok=True)
KEEP_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO('yolov8n.pt')

print(f"🚀 Filtering {len(list(INPUT_DIR.glob('*.jpg')))} images for humans...")

for img_path in INPUT_DIR.glob("*.jpg"):
    try:
        results = model(img_path, verbose=False, conf=0.5)
        
        has_person = False
        for r in results:
            for c in r.boxes.cls:
                if int(c) == 0:
                    has_person = True
                    break
        
        if has_person:
            shutil.move(str(img_path), str(KEEP_DIR / img_path.name))
        else:
            shutil.move(str(img_path), str(JUNK_DIR / img_path.name))
            
    except Exception as e:
        print(f"⚠️ Error reading {img_path.name}: {e}")

print("✅ Filtering Complete.")
print(f"   - Kept images (Has Person) -> {KEEP_DIR}")
print(f"   - Junk images (Empty/Text) -> {JUNK_DIR}")