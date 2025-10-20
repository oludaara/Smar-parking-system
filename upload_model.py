# upload_model.py
# Handles downloading and uploading of YOLOv8 model weights (best.pt)
# Works seamlessly with Supabase and server.py

import os
import requests
from supabase import create_client, Client

# ------------------- SUPABASE CONFIG -------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME", "violations")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing Supabase credentials. Please check your environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ------------------- DOWNLOAD MODEL -------------------
def download_model(model_url=None, local_path="models/best.pt"):
    """
    Downloads the YOLO model (best.pt) from Supabase (or any URL) if not already available.
    """
    os.makedirs("models", exist_ok=True)

    if os.path.exists(local_path):
        print(f"[INFO] Model already exists at {local_path}")
        return local_path

    model_url = model_url or os.getenv("MODEL_URL")
    if not model_url:
        raise ValueError("MODEL_URL not found. Please set it in your environment variables.")

    print(f"[INFO] Downloading model from: {model_url}")
    try:
        r = requests.get(model_url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        print("[INFO] Model download complete.")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        raise e

    return local_path


# ------------------- UPLOAD MODEL -------------------
def upload_model_to_supabase(local_model_path="models/best.pt", folder_name="models"):
    """
    Uploads the local YOLO model to Supabase storage and returns the public URL.
    """
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model not found at {local_model_path}")

    try:
        with open(local_model_path, "rb") as f:
            data = f.read()

        file_key = f"{folder_name}/best.pt"
        supabase.storage.from_(BUCKET_NAME).upload(file_key, data, {"content-type": "application/octet-stream"}, upsert=True)
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_key)

        print(f"[INFO] Model uploaded successfully. Public URL: {public_url}")
        return public_url

    except Exception as e:
        print(f"[ERROR] Supabase upload failed: {e}")
        raise e


# ------------------- MAIN TEST -------------------
if __name__ == "__main__":
    # You can run this manually from your local system:
    # python upload_model.py
    #
    # It will upload your local best.pt to Supabase and print the public URL.
    #
    # NOTE: Ensure SUPABASE_URL, SUPABASE_SERVICE_KEY, and BUCKET_NAME are set before running.

    model_path = "models/best.pt"

    if not os.path.exists(model_path):
        print("[INFO] Model not found locally. Skipping upload.")
    else:
        url = upload_model_to_supabase(model_path)
        print(f"✅ Uploaded model URL:\n{url}")
