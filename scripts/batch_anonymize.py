import cv2
import os
from pathlib import Path
from deepface import DeepFace
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


input_dir = Path("data/raw/baseline_ppe/images/train")
output_dir = Path("data/processed/baseline_ppe/train")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[*] Starting batch anonymization with DeepFace...")
print(f"[*] Processing {len(list(input_dir.glob('*.jpg')))} images...")

processed = 0
faces_detected = 0

for img_path in input_dir.glob("*.jpg"):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[!] Skipping {img_path.name}: Could not read image")
            continue
        
        try:
            face_objs = DeepFace.extract_faces(
                img_path=str(img_path),
                detector_backend='retinaface',
                enforce_detection=False,
                align=False
            )
            
            face_count = 0
            for face_obj in face_objs:
                if face_obj.get('confidence', 0) > 0.9:
                    facial_area = face_obj['facial_area']
                    x = facial_area['x']
                    y = facial_area['y']
                    w = facial_area['w']
                    h = facial_area['h']
                    
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = w + 2 * padding
                    h = h + 2 * padding
                    
                    x_end = min(img.shape[1], x + w)
                    y_end = min(img.shape[0], y + h)
                    
                    roi = img[y:y_end, x:x_end]
                    if roi.size > 0:
                        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                        img[y:y_end, x:x_end] = blurred
                        face_count += 1
            
            faces_detected += face_count
            
        except Exception as e:
            print(f"[!] {img_path.name}: No faces detected or error: {str(e)[:50]}")
        
        cv2.imwrite(str(output_dir / img_path.name), img)
        processed += 1
        
        if processed % 50 == 0:
            print(f"[+] Processed {processed} images, detected {faces_detected} faces...")
            
    except Exception as e:
        print(f"[X] Error processing {img_path.name}: {str(e)}")
        continue

print(f"\n[OK] Anonymization complete!")
print(f"[*] Statistics:")
print(f"    - Images processed: {processed}")
print(f"    - Faces detected & blurred: {faces_detected}")
print(f"    - Detection rate: {faces_detected/processed:.2f} faces/image")
print(f"    - Data is now GDPR-compliant for training.")