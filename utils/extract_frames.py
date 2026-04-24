import cv2
import os

def extract_frames(video_path, output_folder, interval_sec=0.20):   # ← Changed to 0.20
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps * interval_sec))
    
    frame_count = 0
    saved_count = 0
    
    print(f"📹 Processing: {os.path.basename(video_path)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"✅ Saved {saved_count} frames → {output_folder}\n")

# ====================== VIDEO MAPPINGS ======================
video_mappings = [
    ("video3.mp4",  "dataset/raw/phones/iphone"),
    ("video2.mp4",  "dataset/raw/phones/samsung"),
    ("video1.mp4",  "dataset/raw/phones/oppo"),
    ("video.mp4",   "dataset/raw/phones/vivo"),
]

for video_file, out_folder in video_mappings:
    if os.path.exists(video_file):
        extract_frames(video_file, out_folder)
    else:
        print(f"❌ Video not found: {video_file}")

print("🎉 Frame extraction completed!")