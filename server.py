# server.py
# Optimized Flask + YOLOv8 + OCR server for low-memory environments (Render free tier)

import os
import cv2
import numpy as np
import pytesseract
import requests
import torch
from ultralytics import YOLO
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, request, jsonify
from supabase import create_client, Client
import mimetypes
from dotenv import load_dotenv
load_dotenv()

# ------------------- OPTIMIZATIONS -------------------
torch.set_grad_enabled(False)  # Disable autograd globally (saves memory)

# ------------------- FLASK APP -------------------
app = Flask(__name__)

# ------------------- ROUTES -------------------
@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "message": "Smart Parking System Server is running"
    })

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Server is running"})

# ------------------- SETTINGS -------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "cropped_plates"
DEBUG_DIR = "debug_preprocessed"
ANNOTATED_DIR = "annotated"
for d in [UPLOAD_DIR, OUTPUT_DIR, DEBUG_DIR, ANNOTATED_DIR]:
    os.makedirs(d, exist_ok=True)

TESSERACT_CMD = os.getenv("TESSERACT_CMD") or "/usr/bin/tesseract"
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME", "violations")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing Supabase credentials")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ------------------- MODEL -------------------
def download_model():
    """Download YOLO model (best.pt) from Supabase if not present"""
    model_url = os.getenv("MODEL_URL")
    if not model_url:
        raise ValueError("MODEL_URL not found in env vars")

    os.makedirs("models", exist_ok=True)
    local_path = "models/best.pt"
    if not os.path.exists(local_path):
        print(f"[INFO] Downloading YOLO model from {model_url} ...")
        r = requests.get(model_url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        print("[INFO] Model download complete.")
    return local_path

model_path = download_model()
model = YOLO(model_path)
model.fuse()  # fuse Conv+BN layers → less memory

# ------------------- HELPERS -------------------
def run_ocr_on_crop(crop, save_name=None):
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        if save_name:
            cv2.imwrite(os.path.join(DEBUG_DIR, save_name), gray)

        text = pytesseract.image_to_string(
            gray,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        plate_text = ''.join(c for c in text if c.isalnum())
        return plate_text if plate_text else None
    except Exception as e:
        print(f"[ERROR] OCR error: {e}")
        return None

def upload_to_supabase_storage(local_path, public_folder=""):
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        file_key = f"{public_folder}/{Path(local_path).name}"
        content_type, _ = mimetypes.guess_type(str(local_path))
        if not content_type:
            content_type = "image/jpeg"

        supabase.storage.from_(BUCKET_NAME).upload(
            file_key, data, {"content-type": content_type}, upsert=True
        )
        return supabase.storage.from_(BUCKET_NAME).get_public_url(file_key)
    except Exception as e:
        print(f"[ERROR] Supabase upload failed: {e}")
        return None

def insert_plate_record(camera_id, plate_text, confidence, plate_url, scene_url):
    try:
        row = {
            "camera_id": camera_id,
            "plate_text": plate_text,
            "confidence": confidence,
            "plate_url": plate_url,
            "scene_url": scene_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "new",
        }
        supabase.table("violations").insert(row).execute()
        return True
    except Exception as e:
        print(f"[ERROR] Supabase insert exception: {e}")
        return False

# ------------------- FLASK ROUTE -------------------
@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        camera_id = request.form.get("camera_id") or request.headers.get("X-Camera-ID", "CAM1")

        if "image" in request.files:
            img_bytes = request.files["image"].read()
        else:
            img_bytes = request.data
        if not img_bytes:
            return jsonify({"error": "No image data received"}), 400

        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
        save_path = os.path.join(UPLOAD_DIR, filename)
        cv2.imwrite(save_path, img)

        # Run YOLO detection with smaller imgsz to save memory
        results = model.predict(img, conf=0.5, imgsz=320, device="cpu", verbose=False)
        annotated = img.copy()
        detected_plates = []

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = float(confs[i])
                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = model.names[class_ids[i]] if hasattr(model, "names") else str(class_ids[i])
                if class_name.lower() != "plate":
                    continue

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Crop plate
                crop = img[y1:y2, x1:x2]
                crop_name = f"{os.path.splitext(filename)[0]}_plate_{i}.jpg"
                crop_path = os.path.join(OUTPUT_DIR, crop_name)
                cv2.imwrite(crop_path, crop)

                plate_text = run_ocr_on_crop(crop, crop_name)
                plate_url = upload_to_supabase_storage(crop_path, camera_id)

                detected_plates.append({
                    "file": crop_name,
                    "text": plate_text or "unreadable",
                    "plate_url": plate_url,
                    "confidence": confidence
                })

                del crop  # free memory

        # Save annotated scene
        ann_name = f"{os.path.splitext(filename)[0]}_ann.jpg"
        ann_path = os.path.join(ANNOTATED_DIR, ann_name)
        cv2.imwrite(ann_path, annotated)
        scene_url = upload_to_supabase_storage(ann_path, camera_id)

        for dp in detected_plates:
            insert_plate_record(camera_id, dp["text"], dp["confidence"], dp.get("plate_url"), scene_url)

        del img, annotated  # free memory
        cv2.destroyAllWindows()

        return jsonify({
            "status": "ok",
            "file": filename,
            "scene_url": scene_url,
            "plates": detected_plates
        })

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------- MAIN -------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)   # no debug=True → less memory
