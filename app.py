from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
import json
import os
import base64

from classifiers import inception_classifier

app = Flask(__name__)
CORS(app)

HISTORY_FILE = "classification_history.json"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

CLASS_NAME_MAP: dict[str, str] = {
    "vidrio": "Glass",
    "plastico": "Plastic",
    "plastico": "Plastic",
    "metal": "Metal",
    "papel": "Paper",
    "carton": "Cardboard",
    "carton": "Cardboard",
    "organico": "Organic",
    "organico": "Organic",
    "basura": "Trash",
    "bateria": "Battery",
    "bateria": "Battery",
}

def normalize_class(name: str) -> str:
    return CLASS_NAME_MAP.get(name.lower(), name.capitalize())


print("Loading models...")
detector = YOLO(os.path.join(MODELS_DIR, "best_detector.pt"))
classifier = YOLO(os.path.join(MODELS_DIR, "yolo_cls.pt"))

if inception_classifier.is_available():
    print("Advanced classifier loaded.")
    STATIC_MODEL = "inception"
else:
    print("Using default classifier.")
    STATIC_MODEL = "yolo_cls"

print("Ready.")


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_to_history(detections):
    history = load_history()
    for det in detections:
        history.append({
            "id": len(history) + 1,
            "category": det["class"],
            "confidence": det["confidence"],
            "date": datetime.now().isoformat(),
            "timestamp": datetime.now().timestamp(),
        })
    save_history(history)


def classify_crop_yolo(crop_bgr: np.ndarray) -> dict | None:
    results = classifier.predict(source=crop_bgr, verbose=False)
    if not results:
        return None
    probs = results[0].probs
    if probs is None:
        return None
    cls_id = int(probs.top1)
    conf = float(probs.top1conf)
    return {
        "class": normalize_class(classifier.names[cls_id]),
        "confidence": round(conf, 2),
    }


def classify_crop_inception(crop_bgr: np.ndarray) -> dict | None:
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    try:
        result = inception_classifier.predict(pil)
        return {
            "class": result["class"],
            "confidence": round(result["confidence"], 2),
        }
    except Exception as exc:
        print(f"Classifier error: {exc}")
        return None


def run_pipeline(img_bgr: np.ndarray, use_inception: bool = False) -> list[dict]:
    det_results = detector(img_bgr, stream=False, conf=0.4, verbose=False)
    detections = []
    h_img, w_img = img_bgr.shape[:2]

    for r in det_results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            y1c, y2c = max(0, y1), min(h_img, y2)
            x1c, x2c = max(0, x1), min(w_img, x2)
            crop = img_bgr[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            if use_inception and inception_classifier.is_available():
                result = classify_crop_inception(crop)
            else:
                result = classify_crop_yolo(crop)

            if result:
                detections.append({
                    "class": result["class"],
                    "confidence": result["confidence"],
                    "bbox": [x1, y1, x2, y2],
                })

    return detections


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detections = run_pipeline(img, use_inception=(STATIC_MODEL == "inception"))
        if detections:
            add_to_history(detections)
        return jsonify({"detections": detections, "model_used": STATIC_MODEL})
    except Exception as exc:
        print(f"Error in /classify: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/classify-frame", methods=["POST"])
def classify_frame():
    try:
        data = request.get_json()
        if not data or "frame" not in data:
            return jsonify({"error": "No frame data"}), 400
        raw = data["frame"]
        if "," in raw:
            raw = raw.split(",", 1)[1]
        frame_bytes = base64.b64decode(raw)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode frame"}), 400
        detections = run_pipeline(img, use_inception=False)
        return jsonify({"detections": detections})
    except Exception as exc:
        print(f"Error in /classify-frame: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/history", methods=["GET"])
def get_history():
    try:
        return jsonify({"history": load_history()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        history = load_history()
        if not history:
            return jsonify({
                "totalScans": 0,
                "thisWeek": 0,
                "mostCommon": "N/A",
                "categoryDistribution": [],
                "recentClassifications": [],
            })
        total_scans = len(history)
        week_ago = datetime.now().timestamp() - 7 * 24 * 3600
        this_week = sum(1 for h in history if h.get("timestamp", 0) > week_ago)
        categories: dict[str, int] = {}
        for item in history:
            cat = item["category"]
            categories[cat] = categories.get(cat, 0) + 1
        most_common = max(categories, key=categories.get) if categories else "N/A"
        distribution = sorted(
            [{"category": c, "count": n} for c, n in categories.items()],
            key=lambda x: x["count"],
            reverse=True,
        )
        recent = sorted(history, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]
        return jsonify({
            "totalScans": total_scans,
            "thisWeek": this_week,
            "mostCommon": most_common,
            "categoryDistribution": distribution,
            "recentClassifications": recent,
        })
    except Exception as exc:
        print(f"Error in /stats: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/clear-history", methods=["DELETE"])
def clear_history():
    try:
        save_history([])
        return jsonify({"message": "History cleared"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
