"""
╔══════════════════════════════════════════════════════════════════╗
║  Ritians Transport – Face Recognition Service                   ║
║  v1.0 – DeepFace ArcFace + Cosine Similarity + In-Memory Cache ║
╚══════════════════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
import os
import logging
from PIL import Image
from deepface import DeepFace
from pymongo import MongoClient
from datetime import datetime
import threading
import tempfile

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Config ─────────────────────────────────────────────────────────
MONGO_URI         = os.getenv("MONGODB_URI", "mongodb://localhost:27017/ritians")
MODEL_NAME        = "ArcFace"
DETECTOR_BACKEND  = "opencv"
COSINE_THRESHOLD  = 0.40   # lower = stricter (ArcFace cosine distance)
EMBEDDING_DIM     = 512

# ── MongoDB ─────────────────────────────────────────────────────────
client   = MongoClient(MONGO_URI)
db       = client["ritians"]
students = db["students"]
attend   = db["attendance"]

# ── In-Memory Embedding Cache ────────────────────────────────────────
# { regNo: { embedding: np.array, name: str, busNo: str, route: str, ... } }
embedding_cache = {}
cache_lock      = threading.Lock()

def load_embeddings_to_cache():
    """Load all registered face embeddings into RAM at startup."""
    log.info("🔄 Loading face embeddings into cache...")
    count = 0
    for student in students.find({"face_registered": True, "face_embedding": {"$exists": True}}):
        emb = student.get("face_embedding")
        if emb and len(emb) == EMBEDDING_DIM:
            with cache_lock:
                embedding_cache[student["regNo"]] = {
                    "embedding": np.array(emb, dtype=np.float32),
                    "name":      student.get("name", ""),
                    "regNo":     student["regNo"],
                    "busNo":     student.get("busNo", ""),
                    "route":     student.get("route", ""),
                    "boardStop": student.get("boardStop", ""),
                    "department":student.get("department", ""),
                    "year":      student.get("year", ""),
                    "className": student.get("className", ""),
                }
            count += 1
    log.info(f"✅ Loaded {count} embeddings into cache.")

def decode_base64_image(b64_string):
    """Decode base64 image → PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img

def get_embedding(img: Image.Image):
    """Extract ArcFace embedding from PIL image."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = DeepFace.represent(
            img_path      = tmp_path,
            model_name    = MODEL_NAME,
            detector_backend = DETECTOR_BACKEND,
            enforce_detection = True,
        )
        embedding = np.array(result[0]["embedding"], dtype=np.float32)
        return embedding
    finally:
        os.unlink(tmp_path)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (0 = identical, 2 = opposite)."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 2.0
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))

def find_best_match(query_emb: np.ndarray):
    """Search all cached embeddings for best cosine match."""
    best_dist  = float("inf")
    best_match = None
    with cache_lock:
        for reg_no, data in embedding_cache.items():
            dist = cosine_distance(query_emb, data["embedding"])
            if dist < best_dist:
                best_dist  = dist
                best_match = data
    return best_match, best_dist

# ──────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    with cache_lock:
        cached = len(embedding_cache)
    return jsonify({"ok": True, "model": MODEL_NAME, "cached_embeddings": cached})


@app.route("/register-face", methods=["POST"])
def register_face():
    """
    Register a student's face.
    Body: { regNo: str, image: base64_string }
    """
    data   = request.json or {}
    reg_no = data.get("regNo", "").strip().upper()
    image  = data.get("image", "")

    if not reg_no or not image:
        return jsonify({"success": False, "message": "regNo and image are required."}), 400

    # Check student exists
    student = students.find_one({"regNo": reg_no})
    if not student:
        return jsonify({"success": False, "message": f"Student {reg_no} not found in database."}), 404

    # Check duplicate
    if student.get("face_registered"):
        return jsonify({"success": False, "message": "Face already registered. Contact admin to re-register."}), 409

    try:
        img = decode_base64_image(image)
    except Exception as e:
        return jsonify({"success": False, "message": f"Invalid image data: {str(e)}"}), 400

    try:
        embedding = get_embedding(img)
    except Exception as e:
        msg = str(e)
        if "Face could not be detected" in msg or "No face" in msg:
            return jsonify({"success": False, "message": "No face detected. Please ensure proper lighting and face the camera."}), 422
        log.error(f"Embedding error for {reg_no}: {e}")
        return jsonify({"success": False, "message": "Face detection failed. Please try again."}), 500

    # Save to MongoDB
    students.update_one(
        {"regNo": reg_no},
        {"$set": {
            "face_embedding":  embedding.tolist(),
            "face_registered": True,
            "face_registered_at": datetime.utcnow(),
        }}
    )

    # Update in-memory cache immediately
    with cache_lock:
        embedding_cache[reg_no] = {
            "embedding":  embedding,
            "name":       student.get("name", ""),
            "regNo":      reg_no,
            "busNo":      student.get("busNo", ""),
            "route":      student.get("route", ""),
            "boardStop":  student.get("boardStop", ""),
            "department": student.get("department", ""),
            "year":       student.get("year", ""),
            "className":  student.get("className", ""),
        }

    log.info(f"✅ Face registered for {reg_no} – {student.get('name')}")
    return jsonify({"success": True, "message": "Face registered successfully!", "regNo": reg_no})


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Recognize a face from webcam frame.
    Body: { image: base64_string, session: str (morning|evening) }
    Returns matched student info + attendance status.
    """
    data    = request.json or {}
    image   = data.get("image", "")
    session = data.get("session", "morning")

    if not image:
        return jsonify({"success": False, "message": "Image is required."}), 400

    with cache_lock:
        if len(embedding_cache) == 0:
            return jsonify({"success": False, "message": "No registered faces in system."}), 503

    try:
        img = decode_base64_image(image)
    except Exception as e:
        return jsonify({"success": False, "message": f"Invalid image: {str(e)}"}), 400

    try:
        query_emb = get_embedding(img)
    except Exception as e:
        msg = str(e)
        if "Face could not be detected" in msg or "No face" in msg:
            return jsonify({"success": False, "message": "No face detected in frame.", "noFace": True}), 200
        return jsonify({"success": False, "message": "Face detection failed."}), 500

    best_match, dist = find_best_match(query_emb)

    if dist > COSINE_THRESHOLD:
        log.info(f"❌ No match found. Best distance: {dist:.4f}")
        return jsonify({
            "success":    False,
            "message":    "Face not recognized. Please ensure proper lighting.",
            "confidence": round((1 - dist) * 100, 1),
        })

    # Matched!
    reg_no     = best_match["regNo"]
    confidence = round((1 - dist) * 100, 1)

    # Check duplicate attendance today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    existing = attend.find_one({
        "regNo":   reg_no,
        "session": session,
        "timestamp": {"$gte": today_start},
    })

    if existing:
        return jsonify({
            "success":      True,
            "alreadyMarked": True,
            "message":      f"Attendance already marked for {best_match['name']} today.",
            "student":      {k: v for k, v in best_match.items() if k != "embedding"},
            "confidence":   confidence,
        })

    # Mark attendance
    record = {
        "regNo":      reg_no,
        "name":       best_match["name"],
        "busNo":      best_match["busNo"],
        "route":      best_match["route"],
        "boardStop":  best_match["boardStop"],
        "department": best_match["department"],
        "year":       best_match["year"],
        "className":  best_match["className"],
        "session":    session,
        "timestamp":  datetime.utcnow(),
        "confidence": confidence,
        "status":     "present",
    }
    attend.insert_one(record)

    # Update last_attendance on student
    students.update_one({"regNo": reg_no}, {"$set": {"last_attendance": datetime.utcnow()}})

    log.info(f"✅ Attendance marked: {reg_no} – {best_match['name']} | conf={confidence}% | dist={dist:.4f}")
    return jsonify({
        "success":       True,
        "alreadyMarked": False,
        "message":       f"Attendance marked for {best_match['name']}!",
        "student":       {k: v for k, v in best_match.items() if k != "embedding"},
        "confidence":    confidence,
        "timestamp":     record["timestamp"].isoformat(),
    })


@app.route("/reload-cache", methods=["POST"])
def reload_cache():
    """Force reload of embedding cache from MongoDB."""
    with cache_lock:
        embedding_cache.clear()
    load_embeddings_to_cache()
    with cache_lock:
        count = len(embedding_cache)
    return jsonify({"success": True, "message": f"Cache reloaded with {count} embeddings."})


# ── Startup ────────────────────────────────────────────────────────
def warmup_model():
    log.info("🔥 Warming up ArcFace model...")
    try:
        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            dummy.save(tmp.name)
            try:
                DeepFace.represent(
                    img_path=tmp.name,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                )
            except:
                pass
            os.unlink(tmp.name)
        log.info("✅ Model warmed up!")
    except Exception as e:
        log.warning(f"Warmup failed: {e}")

if __name__ == "__main__":
    load_embeddings_to_cache()
    warmup_model()
    port = int(os.getenv("FLASK_PORT", 5001))
    log.info(f"🚀 Face Recognition Service starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
