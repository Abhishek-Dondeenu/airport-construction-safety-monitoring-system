import os
import shutil
from pathlib import Path
from tqdm import tqdm

INPUT_BASE = Path("data/raw/temp_marshalling")
INPUT_IMAGES = INPUT_BASE / "train/images"
INPUT_LABELS = INPUT_BASE / "train/labels"

OUTPUT_BASE = Path("data/processed/marshalling_custom")
OUTPUT_IMAGES = OUTPUT_BASE / "train/images"
OUTPUT_LABELS = OUTPUT_BASE / "train/labels"


CLASS_MAP = {
    0: 11,      
    1: 13,
    2: 12,
    3: 2
}

if OUTPUT_BASE.exists():
    shutil.rmtree(OUTPUT_BASE)
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

print(f"🚀 Harvesting Marshalling Data...")
print(f"   Source: {INPUT_BASE}")
print(f"   Target: {OUTPUT_BASE}")

processed_count = 0
skipped_count = 0

label_files = list(INPUT_LABELS.glob("*.txt"))

for label_file in tqdm(label_files):
    with open(label_file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    has_valid_data = False
    
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        source_id = int(parts[0])
        
        if source_id in CLASS_MAP:
            target_id = CLASS_MAP[source_id]
            parts[0] = str(target_id)
            new_lines.append(" ".join(parts) + "\n")
            has_valid_data = True
            
    if has_valid_data:
        with open(OUTPUT_LABELS / label_file.name, "w") as f:
            f.writelines(new_lines)
            
        image_found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            img_name = label_file.stem + ext
            src_img = INPUT_IMAGES / img_name
            if src_img.exists():
                shutil.copy(src_img, OUTPUT_IMAGES / img_name)
                image_found = True
                break
        
        if image_found:
            processed_count += 1
        else:
            print(f"Warning: Image not found for {label_file.name}")

print(f"\n🎉 SUCCESS!")
print(f"Migrated {processed_count} images.")
print(f"Class IDs remapped correctly (0->11, 1->13, 2->12, 3->2).")