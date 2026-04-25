import cv2
import os
import numpy as np

# 📁 Folders
upload_folder = "uploads"
result_folder = "results"

# 🎬 Output
output_video = "final_clean_video.mp4"

# 📷 Get images
uploads = sorted([
    img for img in os.listdir(upload_folder)
    if img.endswith((".jpg", ".png", ".jpeg"))
])

# 🔍 Find max width & height (to avoid resizing distortion)
max_w, max_h = 0, 0

for img_name in uploads:
    img = cv2.imread(os.path.join(upload_folder, img_name))
    if img is not None:
        h, w = img.shape[:2]
        max_w = max(max_w, w)
        max_h = max(max_h, h)

# 🎥 Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, 24, (max_w, max_h))

# 📌 Function: place image without distortion
def place_on_canvas(img, canvas_w, canvas_h):
    h, w = img.shape[:2]

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    x_offset = (canvas_w - w) // 2
    y_offset = (canvas_h - h) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

    return canvas

# 🔁 Loop
for img_name in uploads:
    upload_path = os.path.join(upload_folder, img_name)
    result_path = os.path.join(result_folder, "result_" + img_name)

    if not os.path.exists(result_path):
        print(f"⚠️ Missing result for {img_name}")
        continue

    img_upload = cv2.imread(upload_path)
    img_result = cv2.imread(result_path)

    if img_upload is None or img_result is None:
        continue

    # 🧱 Place without stretching
    frame1 = place_on_canvas(img_upload, max_w, max_h)
    frame2 = place_on_canvas(img_result, max_w, max_h)

    # 🎬 Show original (1.5 sec)
    for _ in range(36):
        video.write(frame1)

    # 🎬 Show result (2 sec)
    for _ in range(48):
        video.write(frame2)

video.release()

print("✅ Clean video created: final_clean_video.mp4")