import cv2
import numpy as np
from ultralytics import YOLO

# Choose Detection Method: 'hog' for HOG+SVM, 'yolo' for YOLOv8
DETECTION_METHOD = 'yolo'  # Change to 'hog' if needed

# Load YOLOv8 Model (only if using YOLO)
if DETECTION_METHOD == 'yolo':
    model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano for fast performance

# Load HOG (Histogram of Oriented Gradients) Detector (only if using HOG)
if DETECTION_METHOD == 'hog':
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if DETECTION_METHOD == 'hog':
        # HOG+SVM Detection
        boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.1)
        boxes = np.array(boxes)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), weights.tolist(), 0.5, 0.3)

        # Draw bounding boxes
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        num_people = len(indices)

    elif DETECTION_METHOD == 'yolo':
        # YOLOv8 Detection
        results = model(frame)
        num_people = 0  # Count number of detected people

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0].item())  # Class ID

                # Only count "person" class (class_id 0 in COCO dataset)
                if class_id == 0 and confidence > 0.5:
                    num_people += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display People Count
    cv2.putText(frame, f'People Count: {num_people}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show Frame
    cv2.imshow('People Detection', frame)

    # Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()