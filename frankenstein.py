import cv2
import cvzone
import math
from ultralytics import YOLO

detector = YOLO("best_detector.pt") 
classifier = YOLO("best_classifier.pt") 

# cap = cv2.VideoCapture("./GarbageDetector/Media/garbage.mp4")
cap = cv2.VideoCapture(0) 
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("AI Bismillah")

while True:
    success, img = cap.read()
    if not success: break
    
    # detector model to check the bounding box
    det_results = detector(img, stream=True, conf=0.4, verbose=False)
    
    for r in det_results:
        boxes = r.boxes
        for box in boxes:
            # Get the Bounding Box Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Error Check: Ensure box is valid size
            if w < 10 or h < 10: continue

            # Crop the detected item
            crop_img = img[y1:y2, x1:x2]
            
            if crop_img.size == 0: continue
            
            # Classify the obj
            cls_results = classifier.predict(source=crop_img, verbose=False, conf=0.25)
            
            if cls_results and cls_results[0].boxes:
                best_box = cls_results[0].boxes[0] 
                cls_id = int(best_box.cls[0])
                conf = math.ceil((best_box.conf[0] * 100)) / 100
                
                material_name = classifier.names[cls_id]
                
                # Visualize
                if material_name == 'plastic': color = (255, 0, 0)   
                elif material_name == 'paper': color = (0, 255, 0)   
                elif material_name == 'metal': color = (0, 0, 255)   
                else: color = (255, 0, 255)                          
                
                cvzone.cornerRect(img, (x1, y1, w, h), l=30, t=3, colorR=color)
                cvzone.putTextRect(img, f'{material_name} {conf}', 
                                   (max(0, x1), max(35, y1)), 
                                   scale=1.5, thickness=2, 
                                   colorR=color, offset=10)
            else:
                cvzone.cornerRect(img, (x1, y1, w, h), l=30, t=2, colorR=(100,100,100))

    cv2.imshow("AI Bismillah", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()