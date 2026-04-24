import os
import shutil

src = "dataset/raw"
train_dst = "dataset/train"

os.makedirs(train_dst, exist_ok=True)

for brand in os.listdir(src):
    brand_path = os.path.join(src, brand)
    if not os.path.isdir(brand_path):
        continue

    for model in os.listdir(brand_path):
        model_path = os.path.join(brand_path, model)
        if not os.path.isdir(model_path):
            continue

        dst_path = os.path.join(train_dst, model)
        os.makedirs(dst_path, exist_ok=True)

        for img in os.listdir(model_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(model_path, img)
                dst_file = os.path.join(dst_path, img)
                shutil.copy(src_file, dst_file)

print("✅ Dataset flattened into train folder")