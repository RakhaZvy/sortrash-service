# Backend

This is the Flask backend for the waste classification app using YOLO.

## Setup

1. Install Python dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```
   python app.py
   ```

The server will run on http://localhost:5000

## API

- POST /classify: Accepts an image file and returns YOLO detections.

Request: FormData with 'file' key containing the image.

Response: JSON with 'detections' array containing class, confidence, and bbox.
