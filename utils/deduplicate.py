import os
from PIL import Image
import imagehash
import shutil

def deduplicate_folder(folder_path, threshold=5):
    """Remove near-duplicate images using perceptual hash"""
    if not os.path.exists(folder_path):
        return
    
    hashes = {}
    duplicates = 0
    kept = 0
    
    print(f"🔍 Deduplicating: {folder_path}")
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        filepath = os.path.join(folder_path, filename)
        
        try:
            img = Image.open(filepath)
            h = imagehash.phash(img, hash_size=8)   # perceptual hash
            h_str = str(h)
            
            if h_str in hashes:
                # duplicate found
                os.remove(filepath)
                duplicates += 1
            else:
                hashes[h_str] = filename
                kept += 1
        except Exception:
            os.remove(filepath)  # corrupt file
            duplicates += 1
    
    print(f"✅ Kept {kept} unique frames | Removed {duplicates} duplicates\n")

# Run on all raw folders
base_raw = "dataset/raw/phones"
for model in ["iphone", "samsung", "oppo", "vivo"]:
    folder = os.path.join(base_raw, model)
    deduplicate_folder(folder, threshold=5)

print("🎉 Deduplication completed!")