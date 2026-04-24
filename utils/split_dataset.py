import os
import shutil
import random

train_dir = "dataset/train"
val_dir = "dataset/val"

split_ratio = 0.20  # 20% for validation

# Clear old val folder if exists
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)

os.makedirs(val_dir, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".webp")

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path) if img.lower().endswith(valid_ext)]

    if len(images) < 8:   # Increased minimum to avoid too small classes
        print(f"⚠️ Skipping {class_name} (only {len(images)} images)")
        continue

    random.shuffle(images)

    split_count = max(2, int(len(images) * split_ratio))   # At least 2 images in val

    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True)

    moved = 0
    for img in images[:split_count]:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_path, img)
        shutil.move(src, dst)
        moved += 1

    print(f"✅ {class_name}: {moved} images moved to validation (Total: {len(images)})")

print("\n🎯 Dataset split completed successfully!")