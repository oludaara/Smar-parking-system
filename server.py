import os
import threading
import logging
import tempfile
import cv2
import pytesseract
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from ultralytics import YOLO
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
BUCKET_NAME = "violations"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials missing. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

def process_image(file_bytes, filename, camera_id, timestamp):
    """Background task: save locally, run YOLO + OCR, update DB"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        results = model.predict(tmp_path, save=True, conf=0.4)

        img = cv2.imread(tmp_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        plate_text = pytesseract.image_to_string(
            gray,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ).strip()

        data = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "image_path": filename,
            "plate_text": plate_text,
            "status": "new"
        }
        supabase.table("violations").insert(data).execute()
        logger.info(f"[INFO] Processed {filename}, Plate: {plate_text}")
        os.remove(tmp_path)

    except Exception as e:
        logger.error(f"[ERROR] Failed to process {filename}: {e}")

@app.route("/")
def home():
    return jsonify({
        "message": "Smart Parking System Server is running",
        "status": "ok",
        "endpoints": {
            "POST /upload": "Upload an image for processing",
            "GET /violations": "Fetch all violation records"
        }
    })

@app.route("/upload", methods=["POST"])
def upload():
    try:
        image_data = request.data
        camera_id = request.headers.get("X-Camera-ID", "UNKNOWN")
        timestamp = request.headers.get("X-Timestamp", datetime.now().isoformat())
        filename = f"{camera_id}_{timestamp.replace(':','-').replace(' ','_')}.jpg"

        supabase.storage.from_(BUCKET_NAME).upload(filename, image_data, {"content-type": "image/jpeg"})
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)

        threading.Thread(target=process_image, args=(image_data, filename, camera_id, timestamp)).start()

        return jsonify({
            "status": "ok",
            "message": "Image saved to Supabase, processing in background",
            "filename": filename,
            "public_url": public_url
        })

    except Exception as e:
        logger.error(f"[ERROR] Upload failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/violations", methods=["GET"])
def get_violations():
    try:
        response = supabase.table("violations").select("*").order("timestamp", desc=True).execute()
        return jsonify(response.data)
    except Exception as e:
        logger.error(f"[ERROR] Fetching violations failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
