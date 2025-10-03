# server.py
# Flask server for ESP32 uploads:
# - Receives image
# - Runs YOLOv8 detection
# - Crops plate + runs OCR
# - Uploads annotated + cropped plates to Supabase
# - Validates against registered_vehicles DB
# - Inserts violation record with offense type + alert

import os
import cv2
import numpy as np
import pytesseract
import requests
from ultralytics import YOLO
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, request, jsonify
from supabase import create_client, Client
import mimetypes
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# ------------------- ROUTES -------------------
@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "message": "Smart Parking System Server is running",
        "endpoints": {
            "POST /upload": "Upload an image for processing",
            "GET /test": "Test server connectivity"
        }
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

# Configure Tesseract
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or "/usr/bin/tesseract"
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME", "violations")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase credentials missing. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ------------------- MODEL DOWNLOAD -------------------
def download_model():
    model_url = os.getenv("MODEL_URL")
    if not model_url:
        raise ValueError("MODEL_URL not found in environment variables")
    
    os.makedirs("models", exist_ok=True)
    local_path = "models/best.pt"
    
    if not os.path.exists(local_path):
        print(f"[INFO] Downloading YOLO model from {model_url} ...")
        r = requests.get(model_url)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download model: HTTP {r.status_code}")
        with open(local_path, "wb") as f:
            f.write(r.content)
        print("[INFO] Model download complete.")
    
    return local_path

model_path = download_model()
model = YOLO(model_path)

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

        supabase.storage.from_(BUCKET_NAME).upload(file_key, data, {"content-type": content_type}, upsert=True)
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_key)
        if isinstance(public_url, dict):
            return public_url.get("publicURL") or public_url.get("public_url")
        return public_url
    except Exception as e:
        print(f"[ERROR] Supabase upload failed: {e}")
        return None


def insert_plate_record(camera_id, plate_text, confidence, plate_url, scene_url):
    """Check plate validity, decide offense, insert into DB, return alert msg."""
    try:
        norm_plate = (plate_text or "").upper().strip()

        # Validate against registered_vehicles
        res = supabase.table("registered_vehicles").select("plate_text").eq("plate_text", norm_plate).execute()
        registered = len(res.data) > 0

        if registered:
            offense = "Parking Violation"
            alert_msg = f"Wrong parking detected by Camera: {camera_id}"
        else:
            offense = "Unregistered Vehicle + Parking Violation"
            alert_msg = "Invalid Plate Number or Unregistered Vehicle. Committed double offence"

        row = {
            "camera_id": camera_id,
            "plate_text": norm_plate,
            "confidence": confidence,
            "plate_url": plate_url,
            "scene_url": scene_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "offense_type": offense,
            "status": "new",
        }
        supabase.table("violations").insert(row).execute()
        return alert_msg
    except Exception as e:
        print(f"[ERROR] Supabase insert exception: {e}")
        return None

# ------------------- FLASK ROUTE -------------------
@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        camera_id = request.form.get("camera_id") or request.headers.get("X-Camera-ID", "CAM1")
        
        if "image" in request.files:
            img_file = request.files["image"]
            img_bytes = img_file.read()
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

        results = model.predict(img, conf=0.5, imgsz=640, verbose=False)
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

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                crop = img[y1:y2, x1:x2]
                crop_name = f"{os.path.splitext(filename)[0]}_plate_{i}.jpg"
                crop_path = os.path.join(OUTPUT_DIR, crop_name)
                cv2.imwrite(crop_path, crop)

                plate_text = run_ocr_on_crop(crop, crop_name)
                print(f"[OCR] {crop_name} → {plate_text} (conf {confidence:.2f})")

                plate_url = upload_to_supabase_storage(crop_path, camera_id)

                # Insert + validate
                alert_msg = insert_plate_record(camera_id, plate_text, confidence, plate_url, None)

                detected_plates.append({
                    "file": crop_name,
                    "text": plate_text or "unreadable",
                    "plate_url": plate_url,
                    "confidence": confidence,
                    "alert": alert_msg
                })

        # Save annotated scene
        ann_name = f"{os.path.splitext(filename)[0]}_ann.jpg"
        ann_path = os.path.join(ANNOTATED_DIR, ann_name)
        cv2.imwrite(ann_path, annotated)
        scene_url = upload_to_supabase_storage(ann_path, camera_id)

        # Update violations with scene_url (if needed)
        # (already passed as None in insert, you can enhance to update it later)

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
    app.run(host="0.0.0.0", port=5000, debug=True)
