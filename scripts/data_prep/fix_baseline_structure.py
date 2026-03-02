import os
import shutil
from pathlib import Path

BASE_DIR = Path("data/processed/baseline_ppe")

def reorganize_split(split_name):
    old_img_dir = BASE_DIR / split_name
    old_label_dir = BASE_DIR / "labels" / split_name
    
    new_img_dir = BASE_DIR / split_name / "images"
    new_label_dir = BASE_DIR / split_name / "labels"
    
    print(f"🔧 Processing {split_name}...")
    
    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)
    
    for item in os.listdir(old_img_dir):
        src = old_img_dir / item
        dst = new_img_dir / item
        
        if src.is_file():
            shutil.move(src, dst)
            
    if old_label_dir.exists():
        for item in os.listdir(old_label_dir):
            src = old_label_dir / item
            dst = new_label_dir / item
            
            if src.is_file():
                shutil.move(src, dst)
        
        try:
            os.rmdir(old_label_dir)
        except:
            pass
            
    print(f"✅ {split_name} Done! Structure is now Standard YOLO.")

if BASE_DIR.exists():
    reorganize_split("train")
    reorganize_split("val")
    reorganize_split("test")
    
    try:
        os.rmdir(BASE_DIR / "labels")
    except:
        pass
        
    print("\n🎉 SUCCESS: baseline_ppe is now fixed!")
else:
    print(f"❌ Error: Could not find {BASE_DIR}")