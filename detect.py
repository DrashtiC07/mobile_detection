import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import sys

# ========================= CONFIG =========================
device = torch.device("cpu")
class_names = ['iphone', 'oppo', 'samsung', 'vivo']

MODEL_PATH = "models/efficientnet_final_fixed.pth"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ========================= MODEL =========================
model = efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(1280, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(512, 4)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ========================= TRANSFORM =========================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ========================= CLASSIFIER =========================
def classify_phone(cropped_img):
    img_tensor = transform(cropped_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_idx = torch.max(probs, 0)

    return class_names[pred_idx.item()], confidence.item() * 100


# ========================= DETECT FUNCTION =========================
def detect_and_classify(image_name):

    # 👉 Read from uploads folder
    image_path = os.path.join(UPLOAD_FOLDER, image_name)

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Could not read image")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > best_area and area > 5000:
            best_area = area
            x,y,w,h = cv2.boundingRect(cnt)
            best_box = (x, y, w, h)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    if best_box:
        x, y, w, h = best_box

        # 🔥 shrink box
        pad = 10
        x = max(0, x + pad)
        y = max(0, y + pad)
        w = w - 2*pad
        h = h - 2*pad

        cropped = pil_img.crop((x, y, x+w, y+h))
        brand, confidence = classify_phone(cropped)

        # Draw box
        draw.rectangle([x, y, x+w, y+h], outline="lime", width=5)

        label = f"{brand.upper()} ({confidence:.1f}%)"

        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0,0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 👉 TEXT INSIDE BOX
        text_x = x + 5
        text_y = y + 5

        draw.rectangle(
            [text_x, text_y,
             text_x + text_w + 10,
             text_y + text_h + 5],
            fill="lime"
        )

        draw.text(
            (text_x + 5, text_y),
            label,
            fill="black",
            font=font
        )

        print(f"✅ {brand.upper()} ({confidence:.1f}%)")

    else:
        print("⚠️ No phone detected → using full image")
        brand, confidence = classify_phone(pil_img)

        draw.text((20,20),
                  f"{brand.upper()} ({confidence:.1f}%)",
                  fill="lime")

    # 👉 SAVE IN results/
    output_path = os.path.join(RESULT_FOLDER, "result_" + image_name)
    pil_img.save(output_path)

    print(f"📁 Saved to: {output_path}")


# ========================= MAIN =========================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_and_classify(sys.argv[1])
    else:
        print("Usage:")
        print("python detect.py image.jpg")