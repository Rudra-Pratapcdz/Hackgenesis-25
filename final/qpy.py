import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

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

# Global variables for people count and estimated time
num_people = 0
estimated_time = 0

def generate_frames():
    global num_people, estimated_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        num_people = 0  # Reset count per frame

        if DETECTION_METHOD == 'hog':
            # HOG+SVM Detection
            boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.1)
            boxes = np.array(boxes)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), weights.tolist(), 0.5, 0.3)

            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            num_people = len(indices)

        elif DETECTION_METHOD == 'yolo':
            # YOLOv8 Detection
            results = model(frame)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())

                    # Count only "person" class (ID 0 in COCO dataset)
                    if class_id == 0 and confidence > 0.5:
                        num_people += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate Estimated Time
        estimated_time = num_people * 7  # in minutes

        # Display Information
        cv2.putText(frame, f'People Count: {num_people}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Estimated Time: {estimated_time} min', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/run-qpy')
def run_qpy():
    # Logic to run when the Q-watch card is clicked
    try:
        # Execute qpy.py script from terminal
        subprocess.Popen(['python', 'qpy.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify(message="qpy.py script is running!")
    except Exception as e:
        return jsonify(message=f"Error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    return jsonify({'num_people': num_people, 'estimated_time': estimated_time})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Ensures the app runs on port 5000
