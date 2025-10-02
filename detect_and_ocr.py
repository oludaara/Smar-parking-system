# detect_and_ocr.py
# - Loads YOLOv8 model
# - Runs inference on images
# - Crops detected license plates
# - Runs Tesseract OCR on each crop
# - Prints recognized plate text

import os
import cv2
import pytesseract
from ultralytics import YOLO

# ------------------- SETTINGS -------------------
MODEL_PATH = "best.pt"              # your trained YOLO weights
SOURCE = "test_images/"             # folder with car images
OUTPUT_DIR = "cropped_plates"       # folder to save cropped plates
DEBUG_DIR = "debug_preprocessed"    # to inspect OCR input

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ------------------- INIT -------------------
# Load YOLO model
model = YOLO(MODEL_PATH)

# OCR function
def run_ocr_on_crop(crop, save_name=None):
    # Preprocess for OCR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]
    gray = cv2.medianBlur(gray, 3)

    # Optional: save preprocessed image for debugging
    if save_name:
        cv2.imwrite(os.path.join(DEBUG_DIR, save_name), gray)

    # OCR
    text = pytesseract.image_to_string(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    plate_text = ''.join(c for c in text if c.isalnum())

    return plate_text if plate_text else None

# ------------------- MAIN -------------------
# Run YOLO inference
results = model.predict(source=SOURCE, conf=0.5, imgsz=640)

for r in results:
    img = r.orig_img.copy()
    boxes = r.boxes.xyxy.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)

    filename = os.path.basename(r.path)  # original image name

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[class_ids[i]]

        # Only process plates (make sure 'plate' is in your dataset labels)
        if class_name.lower() == "plate":
            crop = img[y1:y2, x1:x2]

            save_name = f"{os.path.splitext(filename)[0]}_plate_{i}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, crop)

            # Run OCR immediately
            plate_text = run_ocr_on_crop(crop, save_name)
            print(f"{save_path} → {plate_text}")
