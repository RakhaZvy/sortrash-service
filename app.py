from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import math
from datetime import datetime
import json
import os
import base64

app = Flask(__name__)
CORS(app)

# File to store classification history
HISTORY_FILE = 'classification_history.json'

# --- LOAD YOUR MODELS ---
# Make sure these two files are inside your 'backend' folder!
print("Loading models...")
detector = YOLO("best_detector.pt") 
classifier = YOLO("best_classifier.pt")
print("Models loaded!")

def load_history():
    """Load classification history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    """Save classification history to JSON file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def add_to_history(detections):
    """Add new classification to history"""
    history = load_history()
    
    for detection in detections:
        entry = {
            'id': len(history) + 1,
            'category': detection['class'],
            'confidence': detection['confidence'],
            'date': datetime.now().isoformat(),
            'timestamp': datetime.now().timestamp()
        }
        history.append(entry)
    
    save_history(history)

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # 1. Read Image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = np.array(image)
            # Convert RGB to BGR (OpenCV format)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 2. Run Detector (Stage 1)
            # This finds WHERE the trash is
            det_results = detector(img, stream=False, conf=0.4, verbose=False)
            
            detections = []

            for r in det_results:
                boxes = r.boxes
                for box in boxes:
                    # Get Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    
                    # Filter small boxes (noise)
                    if w < 10 or h < 10: 
                        continue

                    # 3. Crop the detected item
                    # Ensure coordinates are within image bounds
                    h_img, w_img, _ = img.shape
                    y1_c, y2_c = max(0, y1), min(h_img, y2)
                    x1_c, x2_c = max(0, x1), min(w_img, x2)
                    
                    crop_img = img[y1_c:y2_c, x1_c:x2_c]
                    
                    if crop_img.size == 0: 
                        continue
                    
                    # 4. Run Classifier (Stage 2)
                    # This finds WHAT TYPE of trash it is (Plastic, Metal, etc.)
                    cls_results = classifier.predict(source=crop_img, verbose=False, conf=0.25)
                    
                    if cls_results and cls_results[0].boxes:
                        best_box = cls_results[0].boxes[0]
                        cls_id = int(best_box.cls[0])
                        # Get confidence score (e.g., 0.85)
                        conf = float(math.ceil((best_box.conf[0] * 100)) / 100)
                        
                        # Get class name (e.g., "plastic")
                        material_name = classifier.names[cls_id]

                        detections.append({
                            'class': material_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })

            # If no trash detected by the two-stage process, return empty
            # Save to history
            if detections:
                add_to_history(detections)

            return jsonify({'detections': detections})

        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/classify-frame', methods=['POST'])
def classify_frame():
    """Real-time frame classification for live webcam tracking"""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400
        
        # Decode base64 image
        frame_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        frame_bytes = base64.b64decode(frame_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Run detector
        det_results = detector(img, stream=False, conf=0.4, verbose=False)
        
        detections = []
        
        for r in det_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                if w < 10 or h < 10: 
                    continue
                
                # Crop the detected item
                h_img, w_img, _ = img.shape
                y1_c, y2_c = max(0, y1), min(h_img, y2)
                x1_c, x2_c = max(0, x1), min(w_img, x2)
                
                crop_img = img[y1_c:y2_c, x1_c:x2_c]
                
                if crop_img.size == 0: 
                    continue
                
                # Run classifier
                cls_results = classifier.predict(source=crop_img, verbose=False, conf=0.25)
                
                if cls_results and cls_results[0].boxes:
                    best_box = cls_results[0].boxes[0]
                    cls_id = int(best_box.cls[0])
                    conf = float(math.ceil((best_box.conf[0] * 100)) / 100)
                    
                    material_name = classifier.names[cls_id]
                    
                    detections.append({
                        'class': material_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return jsonify({'detections': detections})
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get classification history"""
    try:
        history = load_history()
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error loading history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    try:
        history = load_history()
        
        if not history:
            return jsonify({
                'totalScans': 0,
                'thisWeek': 0,
                'mostCommon': 'N/A',
                'categoryDistribution': [],
                'recentClassifications': []
            })
        
        # Calculate total scans
        total_scans = len(history)
        
        # Calculate this week's scans
        now = datetime.now()
        week_ago = now.timestamp() - (7 * 24 * 60 * 60)
        this_week = len([h for h in history if h.get('timestamp', 0) > week_ago])
        
        # Find most common category
        categories = {}
        for item in history:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        most_common = max(categories.items(), key=lambda x: x[1])[0] if categories else 'N/A'
        
        # Category distribution
        category_distribution = [
            {'category': cat, 'count': count}
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Recent classifications (last 10)
        recent = sorted(history, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
        
        return jsonify({
            'totalScans': total_scans,
            'thisWeek': this_week,
            'mostCommon': most_common,
            'categoryDistribution': category_distribution,
            'recentClassifications': recent
        })
        
    except Exception as e:
        print(f"Error calculating stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['DELETE'])
def clear_history():
    """Clear all classification history"""
    try:
        save_history([])
        return jsonify({'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)