import cv2
from PIL import Image
from detect import classify_phone
# ... (same model loading as above)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to PIL and classify
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    brand, conf = classify_phone(pil_img)   # Use classify_phone from detect.py
    
    cv2.putText(frame, f"{brand.upper()} {conf:.1f}%", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Mobile Brand Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()