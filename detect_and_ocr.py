# detect_and_ocr.py
# Handles YOLO detection and OCR reading for number plates
# This module is imported by server.py

import os
import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO

# ------------------- LOAD MODEL -------------------
def load_model(model_path="models/best.pt"):
    """
    Loads and fuses the YOLOv8 model for optimized inference.
    """
    model = YOLO(model_path)
    model.fuse()
    return model

# ------------------- OCR FUNCTION -------------------
def run_ocr_on_crop(crop, debug_path=None):
    """
    Perform OCR on the cropped number plate image using Tesseract.
    Cleans the text and returns only alphanumeric characters.
    """
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)

        if debug_path:
            cv2.imwrite(debug_path, gray)

        text = pytesseract.image_to_string(
            gray,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        plate_text = ''.join(c for c in text if c.isalnum())
        return plate_text if plate_text else None
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return None

# ------------------- DETECTION FUNCTION -------------------
def detect_plates(model, img, save_dir="output", filename_prefix="detected"):
    """
    Runs YOLO detection on an image, extracts plates, and runs OCR.
    Returns list of dictionaries with detected text, confidence, and cropped image paths.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        results = model.predict(img, conf=0.5, imgsz=320, device="cpu", verbose=False)

        detections = []
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(confs[i])
                class_name = model.names[class_ids[i]]

                if class_name.lower() != "plate":
                    continue

                crop = img[y1:y2, x1:x2]
                crop_name = f"{filename_prefix}_plate_{i}.jpg"
                crop_path = os.path.join(save_dir, crop_name)
                cv2.imwrite(crop_path, crop)

                text = run_ocr_on_crop(crop)
                detections.append({
                    "file": crop_name,
                    "confidence": conf,
                    "plate_text": text or "unreadable",
                    "crop_path": crop_path
                })

        return detections

    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return []
