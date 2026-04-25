"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Ritians Transport – Face Recognition Service                           ║
║  v3.0 – Sub-1-Second Pipeline                                           ║
║                                                                         ║
║  KEY UPGRADE vs v2: DeepFace/ArcFace → InsightFace buffalo_sc           ║
║                                                                         ║
║  Why InsightFace is faster:                                             ║
║    • Uses ONNX Runtime (not TensorFlow) → 3-5x faster CPU inference    ║
║    • buffalo_sc = SCRFD face detector + MBF (MobileNet) recognizer     ║
║    • SCRFD detector: ~30ms  vs YuNet ~80ms  vs OpenCV ~400ms           ║
║    • MBF recognizer: ~80ms  vs ArcFace ~400ms on CPU                   ║
║    • No Keras/TF graph overhead, no warmup quirks                      ║
║                                                                         ║
║  Expected latency breakdown (CPU, warm path):                           ║
║    Decode + resize        :  ~10–20  ms                                 ║
║    SCRFD face detection   :  ~25–50  ms                                 ║
║    MBF embedding          :  ~80–150 ms                                 ║
║    FAISS search           :  ~1–3    ms                                 ║
║    Flask + JSON overhead  :  ~5–10   ms                                 ║
║    ──────────────────────────────────────────────────────               ║
║    Total (warm path)      :  ~120–230 ms  🚀 well under 1 second       ║
║                                                                         ║
║  If you MUST keep ArcFace: set USE_INSIGHTFACE=0 in env.               ║
║  The file falls back to the v2 DeepFace path automatically.            ║
║                                                                         ║
║  Install:                                                               ║
║    pip install flask flask-cors pymongo insightface onnxruntime \       ║
║                faiss-cpu pillow numpy opencv-python-headless            ║
║                                                                         ║
║  Optional (keep v2 fallback working):                                   ║
║    pip install deepface tf-keras                                        ║
║                                                                         ║
║  Production run:                                                        ║
║    gunicorn -w 2 -k gthread --threads 4 face_recognition_service:app   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
 
from __future__ import annotations
 
import base64
import io
import logging
import os
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
 
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from pymongo import MongoClient
 
# ══════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
 
# ══════════════════════════════════════════════════════════════════════════
# App + CORS
# ══════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)
 
# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════
MONGO_URI        = os.getenv("MONGODB_URI", "mongodb://localhost:27017/ritians")
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.40"))
MAX_FRAME_WIDTH  = int(os.getenv("MAX_FRAME_WIDTH", "640"))
THREAD_WORKERS   = int(os.getenv("THREAD_WORKERS", "4"))
 
# ── InsightFace config ─────────────────────────────────────────────────────
# buffalo_sc  = SCRFD detector (fast) + MobileNet recognizer (fast, good accuracy)
# buffalo_l   = SCRFD detector       + ResNet100 recognizer (slower, best accuracy)
# antelopev2  = SCRFD 10g (very fast) + ResNet100 (best, requires download)
INSIGHTFACE_MODEL   = os.getenv("INSIGHTFACE_MODEL", "buffalo_sc")
USE_INSIGHTFACE     = os.getenv("USE_INSIGHTFACE", "1") == "1"
EMBEDDING_DIM       = 512   # both ArcFace and MBF output 512-d
 
# ── DeepFace fallback config (used only when USE_INSIGHTFACE=0) ────────────
MODEL_NAME       = "ArcFace"
DETECTOR_BACKEND = os.getenv("FACE_DETECTOR", "yunet")
 
# ══════════════════════════════════════════════════════════════════════════
# Optional imports (graceful fallback)
# ══════════════════════════════════════════════════════════════════════════
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    log.warning("faiss-cpu not installed – using numpy search. Run: pip install faiss-cpu")
 
# ══════════════════════════════════════════════════════════════════════════
# MongoDB
# ══════════════════════════════════════════════════════════════════════════
_mongo_client = MongoClient(MONGO_URI)
_db           = _mongo_client["test"]
students      = _db["students"]
attend        = _db["attendance"]
 
# ══════════════════════════════════════════════════════════════════════════
# InsightFace Model Singleton  (fast ONNX path)
# ══════════════════════════════════════════════════════════════════════════
_insightface_lock = threading.Lock()
_insightface_app  = None   # insightface.app.FaceAnalysis instance
 
def _get_insightface():
    """
    Load InsightFace FaceAnalysis once, reuse forever.
 
    FaceAnalysis bundles:
      • SCRFD face detector    → ~25–50 ms/frame on CPU
      • MBF/ResNet recognizer  → ~80–150 ms/face on CPU
      • Landmark + attribute models (we disable what we don't need)
 
    det_size=(320, 320) is the key CPU speedup vs default (640, 640):
      • 4× fewer pixels fed to detector
      • Detection time halves (~25 ms vs ~50 ms)
      • Still reliable for faces ≥ 80px wide in a 640-wide input frame
    """
    global _insightface_app
    if _insightface_app is not None:
        return _insightface_app
    with _insightface_lock:
        if _insightface_app is None:
            try:
                import insightface
                from insightface.app import FaceAnalysis
                log.info(f"🔨 Loading InsightFace model: {INSIGHTFACE_MODEL} …")
                fa = FaceAnalysis(
                    name       = INSIGHTFACE_MODEL,
                    providers  = ["CPUExecutionProvider"],  # add "CUDAExecutionProvider" if GPU
                )
                # det_size controls the detector input resolution.
                # Smaller = faster; (320,320) is the sweet spot for CPU.
                fa.prepare(ctx_id=0, det_size=(320, 320))
                _insightface_app = fa
                log.info("✅ InsightFace ready.")
            except ImportError:
                log.error("InsightFace not installed! Run: pip install insightface onnxruntime")
                raise
    return _insightface_app
 
# ══════════════════════════════════════════════════════════════════════════
# DeepFace fallback singleton  (v2 path, used only if USE_INSIGHTFACE=0)
# ══════════════════════════════════════════════════════════════════════════
_deepface_lock    = threading.Lock()
_deepface_built   = False
 
def _ensure_deepface_built():
    global _deepface_built
    if _deepface_built:
        return
    with _deepface_lock:
        if not _deepface_built:
            from deepface import DeepFace
            log.info("🔨 Building ArcFace model (fallback)…")
            DeepFace.build_model(MODEL_NAME)
            _deepface_built = True
            log.info("✅ ArcFace model ready.")
 
# ══════════════════════════════════════════════════════════════════════════
# Thread Pool
# ══════════════════════════════════════════════════════════════════════════
_executor = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
 
# ══════════════════════════════════════════════════════════════════════════
# FAISS / NumPy Embedding Index  (unchanged from v2 — already optimal)
# ══════════════════════════════════════════════════════════════════════════
 
class EmbeddingIndex:
    """
    Thread-safe face embedding index.
    FAISS IndexFlatIP on L2-normalised vectors = cosine similarity search.
    Falls back to numpy matmul if faiss-cpu not installed.
    """
 
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim   = dim
        self._lock = threading.RLock()
        self._regnos: list[str]       = []
        self._data:   dict[str, dict] = {}
 
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)
            log.info("🔍 EmbeddingIndex: FAISS IndexFlatIP")
        else:
            self._matrix: Optional[np.ndarray] = None
            log.info("🔍 EmbeddingIndex: numpy fallback")
 
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v
 
    def add(self, reg_no: str, embedding: np.ndarray, info: dict) -> None:
        normed = self._normalize(embedding).astype(np.float32)
        with self._lock:
            if reg_no in self._data:
                self._rebuild_unlocked(
                    [(r, self._data[r]["_raw_emb"], self._data[r])
                     for r in self._regnos if r != reg_no]
                    + [(reg_no, embedding, info)]
                )
            else:
                self._data[reg_no] = {**info, "_raw_emb": embedding}
                self._regnos.append(reg_no)
                if FAISS_AVAILABLE:
                    self._index.add(normed.reshape(1, self.dim))
                else:
                    row = normed.reshape(1, self.dim)
                    self._matrix = row if self._matrix is None \
                                       else np.vstack([self._matrix, row])
 
    def search(self, query: np.ndarray) -> Tuple[Optional[dict], float]:
        normed = self._normalize(query).astype(np.float32)
        with self._lock:
            if not self._regnos:
                return None, 2.0
            if FAISS_AVAILABLE:
                scores, indices = self._index.search(normed.reshape(1, self.dim), 1)
                score = float(scores[0][0])
                idx   = int(indices[0][0])
                if idx < 0:
                    return None, 2.0
                reg_no = self._regnos[idx]
            else:
                sims   = (normed @ self._matrix.T).flatten()
                idx    = int(np.argmax(sims))
                score  = float(sims[idx])
                reg_no = self._regnos[idx]
 
            dist = 1.0 - score
            info = {k: v for k, v in self._data[reg_no].items() if k != "_raw_emb"}
            return info, dist
 
    def rebuild(self, student_docs) -> int:
        entries = []
        for s in student_docs:
            emb = s.get("face_embedding")
            if emb and len(emb) == self.dim:
                entries.append((s["regNo"], np.array(emb, dtype=np.float32), _student_info(s)))
        with self._lock:
            self._rebuild_unlocked(entries)
        return len(entries)
 
    def size(self) -> int:
        with self._lock:
            return len(self._regnos)
 
    def _rebuild_unlocked(self, entries: list) -> None:
        self._regnos = []
        self._data   = {}
        if FAISS_AVAILABLE:
            self._index.reset()
        else:
            self._matrix = None
        for reg_no, embedding, info in entries:
            normed = self._normalize(embedding).astype(np.float32)
            self._data[reg_no] = {**info, "_raw_emb": embedding}
            self._regnos.append(reg_no)
            if FAISS_AVAILABLE:
                self._index.add(normed.reshape(1, self.dim))
            else:
                row = normed.reshape(1, self.dim)
                self._matrix = row if self._matrix is None \
                                   else np.vstack([self._matrix, row])
 
 
face_index = EmbeddingIndex()
 
# ══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════
 
def _student_info(doc: dict) -> dict:
    return {
        "name":       doc.get("name", ""),
        "regNo":      doc["regNo"],
        "busNo":      doc.get("busNo", ""),
        "route":      doc.get("route", ""),
        "boardStop":  doc.get("boardStop", ""),
        "department": doc.get("department", ""),
        "year":       doc.get("year", ""),
        "className":  doc.get("className", ""),
    }
 
 
def decode_base64_image(b64_string: str) -> np.ndarray:
    """
    Decode base64 → BGR numpy array directly.
 
    v3 change: returns BGR ndarray (not PIL Image) to avoid a PIL→ndarray
    conversion step inside get_embedding. InsightFace expects BGR ndarrays.
    DeepFace fallback also works with BGR ndarrays.
    Saves ~5–10 ms per request.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    nparr     = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None – invalid image bytes")
    return bgr
 
 
def resize_if_large(bgr: np.ndarray, max_width: int = MAX_FRAME_WIDTH) -> np.ndarray:
    """Shrink wide frames before detection. Preserves aspect ratio."""
    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr
    scale = max_width / w
    return cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)
 
 
# ══════════════════════════════════════════════════════════════════════════
# Embedding extraction  ← THE CRITICAL HOT PATH
# ══════════════════════════════════════════════════════════════════════════
 
def get_embedding_insightface(bgr: np.ndarray) -> np.ndarray:
    """
    Extract face embedding using InsightFace (ONNX Runtime).
 
    Speed breakdown on CPU:
      SCRFD detection  (320×320 input): ~25–50  ms
      MBF recognition  (112×112 crop):  ~80–150 ms
      Total:                            ~105–200 ms  ✅
 
    InsightFace returns a list of Face objects; we take the one with the
    highest detection score (most confident / largest face in frame).
    """
    fa    = _get_insightface()
    bgr   = resize_if_large(bgr)
    faces = fa.get(bgr)
 
    if not faces:
        raise ValueError("Face could not be detected in the image.")
 
    # If multiple faces, pick the highest-confidence detection
    best_face = max(faces, key=lambda f: f.det_score)
    return best_face.normed_embedding.astype(np.float32)  # already L2-normed
 
 
def get_embedding_deepface(bgr: np.ndarray) -> np.ndarray:
    """
    Fallback: DeepFace/ArcFace embedding (v2 path).
    Used only when USE_INSIGHTFACE=0.
    """
    from deepface import DeepFace
    _ensure_deepface_built()
    bgr = resize_if_large(bgr)
    result = DeepFace.represent(
        img_path         = bgr,
        model_name       = MODEL_NAME,
        detector_backend = DETECTOR_BACKEND,
        enforce_detection= True,
    )
    return np.array(result[0]["embedding"], dtype=np.float32)
 
 
def get_embedding(bgr: np.ndarray) -> np.ndarray:
    """Route to InsightFace (fast) or DeepFace (fallback)."""
    if USE_INSIGHTFACE:
        return get_embedding_insightface(bgr)
    return get_embedding_deepface(bgr)
 
 
# ══════════════════════════════════════════════════════════════════════════
# Index loader
# ══════════════════════════════════════════════════════════════════════════
 
def load_embeddings_to_index() -> int:
    log.info("🔄 Loading face embeddings from MongoDB…")
    cursor = students.find(
        {"face_registered": True, "face_embedding": {"$exists": True}},
        {"regNo": 1, "name": 1, "busNo": 1, "route": 1, "boardStop": 1,
         "department": 1, "year": 1, "className": 1, "face_embedding": 1}
    )
    count = face_index.rebuild(cursor)
    log.info(f"✅ Loaded {count} embeddings.")
    return count
 
# ══════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok":               True,
        "engine":           "InsightFace" if USE_INSIGHTFACE else "DeepFace",
        "model":            INSIGHTFACE_MODEL if USE_INSIGHTFACE else MODEL_NAME,
        "detector":         "SCRFD" if USE_INSIGHTFACE else DETECTOR_BACKEND,
        "faiss":            FAISS_AVAILABLE,
        "cached_embeddings": face_index.size(),
    })
 
 
@app.route("/register-face", methods=["POST"])
def register_face():
    """
    Register a student's face embedding.
    Body: { regNo: str, image: base64_string }
    """
    data   = request.json or {}
    reg_no = data.get("regNo", "").strip().upper()
    image  = data.get("image", "")
 
    if not reg_no or not image:
        return jsonify({"success": False, "message": "regNo and image are required."}), 400
 
    student = students.find_one({"regNo": reg_no})
    if not student:
        return jsonify({"success": False,
                        "message": f"Student {reg_no} not found."}), 404
 
    if student.get("face_registered"):
        return jsonify({"success": False,
                        "message": "Face already registered. Contact admin to re-register."}), 409
 
    try:
        bgr = decode_base64_image(image)
    except Exception as e:
        return jsonify({"success": False, "message": f"Invalid image: {e}"}), 400
 
    try:
        embedding = get_embedding(bgr)
    except Exception as e:
        msg = str(e)
        if "Face could not be detected" in msg or "No face" in msg:
            return jsonify({"success": False,
                            "message": "No face detected. Ensure good lighting and face the camera."}), 422
        log.exception(f"Embedding error for {reg_no}")
        return jsonify({"success": False, "message": "Face detection failed. Try again."}), 500
 
    students.update_one(
        {"regNo": reg_no},
        {"$set": {
            "face_embedding":     embedding.tolist(),
            "face_registered":    True,
            "face_registered_at": datetime.utcnow(),
        }}
    )
 
    info = _student_info(student)
    info["regNo"] = reg_no
    face_index.add(reg_no, embedding, info)
 
    log.info(f"✅ Face registered: {reg_no} – {student.get('name')}")
    return jsonify({"success": True, "message": "Face registered successfully!", "regNo": reg_no})
 
 
@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Recognize a face from a webcam frame.
    Body: { image: base64_string, session: "morning"|"evening" }
    """
    data    = request.json or {}
    image   = data.get("image", "")
    session = data.get("session", "morning")
 
    if not image:
        return jsonify({"success": False, "message": "Image is required."}), 400
 
    if face_index.size() == 0:
        return jsonify({"success": False, "message": "No registered faces in system."}), 503
 
    # Decode in request thread (fast: ~10ms)
    try:
        bgr = decode_base64_image(image)
    except Exception as e:
        return jsonify({"success": False, "message": f"Invalid image: {e}"}), 400
 
    # Embed + search in thread pool (non-blocking)
    def _embed_and_search():
        query_emb = get_embedding(bgr)
        return face_index.search(query_emb)
 
    future = _executor.submit(_embed_and_search)
    try:
        best_match, dist = future.result(timeout=10)
    except Exception as e:
        msg = str(e)
        if "Face could not be detected" in msg or "No face" in msg:
            return jsonify({"success": False,
                            "message": "No face detected in frame.",
                            "noFace": True}), 200
        log.exception("Embedding/search failed")
        return jsonify({"success": False, "message": "Face detection failed."}), 500
 
    confidence = round((1.0 - dist) * 100, 1)
    if dist > COSINE_THRESHOLD:
        log.info(f"❌ No match. dist={dist:.4f}")
        return jsonify({
            "success":    False,
            "message":    "Face not recognized. Ensure good lighting.",
            "confidence": confidence,
        })
 
    reg_no = best_match["regNo"]
 
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    existing = attend.find_one({
        "regNo":     reg_no,
        "session":   session,
        "timestamp": {"$gte": today_start},
    })
 
    if existing:
        return jsonify({
            "success":       True,
            "alreadyMarked": True,
            "message":       f"Attendance already marked for {best_match['name']} today.",
            "student":       best_match,
            "confidence":    confidence,
        })
 
    now = datetime.utcnow()
    record = {
        **best_match,
        "session":    session,
        "timestamp":  now,
        "confidence": confidence,
        "status":     "present",
    }
    attend.insert_one(record)
    students.update_one({"regNo": reg_no}, {"$set": {"last_attendance": now}})
 
    log.info(f"✅ Attendance: {reg_no} – {best_match['name']} | "
             f"conf={confidence}% | dist={dist:.4f} | session={session}")
 
    return jsonify({
        "success":       True,
        "alreadyMarked": False,
        "message":       f"Attendance marked for {best_match['name']}!",
        "student":       best_match,
        "confidence":    confidence,
        "timestamp":     now.isoformat(),
    })
 
 
@app.route("/reload-cache", methods=["POST"])
def reload_cache():
    count = load_embeddings_to_index()
    return jsonify({"success": True,
                    "message": f"Index reloaded with {count} embeddings."})
 
 
# ══════════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════════
 
def _startup():
    if USE_INSIGHTFACE:
        _get_insightface()       # download + load ONNX model once
    else:
        _ensure_deepface_built()
    load_embeddings_to_index()
 
 
if __name__ == "__main__":
    _startup()
    port = int(os.getenv("FLASK_PORT", 5001))
    log.info(f"🚀 Face Recognition Service v3 (sub-1s) starting on :{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
else:
    _startup()
