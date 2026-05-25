# SorTrash Service

Flask backend for the SorTrash waste classification app. Handles image classification and real-time webcam frame detection.

## Requirements

- Python 3.10+
- The model files inside `models/` (committed to the repo)

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Server runs on `http://localhost:5000`.

## API

### `POST /classify`
Upload an image and get waste classification results.

**Request:** `multipart/form-data` with a `file` field containing the image.

**Response:**
```json
{
  "detections": [
    { "class": "Plastic", "confidence": 0.91, "bbox": [x1, y1, x2, y2] }
  ],
  "model_used": "yolo_cls"
}
```

### `POST /classify-frame`
Send a base64-encoded webcam frame for real-time detection.

**Request:**
```json
{ "frame": "<base64 JPEG string>" }
```

**Response:** Same shape as `/classify`.

### `GET /history`
Returns all past classifications.

### `GET /stats`
Returns dashboard stats: total scans, this week count, most common category, and distribution.

### `DELETE /clear-history`
Clears all stored classification history.

## Deployment

The repo includes a `Dockerfile` and `render.yaml` for one-click deploy on [Render.com](https://render.com).

Set the `PORT` environment variable if your host requires a specific port (Render sets this automatically).
