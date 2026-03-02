import os
import shutil
from pathlib import Path

INPUT_LABELS = Path("data/processed/temp_roboflow/valid/labels") 
INPUT_IMAGES = Path("data/processed/temp_roboflow/valid/images")

OUTPUT_LABELS = Path("data/processed/airport_custom/val/labels")
OUTPUT_IMAGES = Path("data/processed/airport_custom/val/images")

CLASS_MAP = {
    1: 11, # Ear-protection -> NEW Class 11
    4: 0,  # Helmet         -> Helmet
    7: 2,  # Vest           -> Vest
    6: 6,  # Person         -> Person
    0: 3,  # Boots          -> Boots
    3: 1,  # Glove          -> Gloves
    2: 4,  # Glass          -> Goggles
}

os.makedirs(OUTPUT_LABELS, exist_ok=True)
os.makedirs(OUTPUT_IMAGES, exist_ok=True)

print(f"🚀 Starting Smart Harvest with correct IDs...")
print(f"[*] Target Class 11 (Ear-Defenders) configured.")

processed_count = 0
files_copied = 0

for label_file in INPUT_LABELS.glob("*.txt"):
    with open(label_file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    has_target_class = False 
    
    keep_image = False

    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        source_id = int(parts[0])
        
        if source_id in CLASS_MAP:
            target_id = CLASS_MAP[source_id]
            
            if target_id == 11:
                keep_image = True
            
            parts[0] = str(target_id)
            new_lines.append(" ".join(parts) + "\n")
            
    if new_lines:
        with open(OUTPUT_LABELS / label_file.name, "w") as f:
            f.writelines(new_lines)
            
        img_name = label_file.stem + ".jpg"
        src_img = INPUT_IMAGES / img_name
        if not src_img.exists():
             src_img = INPUT_IMAGES / (label_file.stem + ".jpeg")
        
        if src_img.exists():
            shutil.copy(src_img, OUTPUT_IMAGES / src_img.name)
            files_copied += 1
    
    processed_count += 1

print(f"✅ Success! Mapped and merged {files_copied} images.")
print(f"   Ear-Defenders are now Class 11.")