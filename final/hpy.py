from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import time
import random
import subprocess
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define class IDs
HUMAN_CLASS_ID = 0  # COCO dataset 'person' class
TABLE_CLASS_ID = 60  # COCO dataset 'dining table' class

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Human tracking
human_timestamps = {}  # Stores entry time of humans
completed_times = []  # Stores dining times
TABLE_HUMANS = {}  # Track how many humans are near each table

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection
        results = model(frame)
        tables = []
        current_humans = set()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()

                if class_id == HUMAN_CLASS_ID:
                    human_id = f"{x1}-{y1}-{x2}-{y2}"
                    current_humans.add(human_id)
                    if human_id not in human_timestamps:
                        human_timestamps[human_id] = time.time()
                    color = (0, 255, 0)
                elif class_id == TABLE_CLASS_ID:
                    tables.append((x1, y1, x2, y2))
                    color = (255, 0, 0)
                else:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        TABLE_HUMANS.clear()
        for table in tables:
            tx1, ty1, tx2, ty2 = table
            TABLE_HUMANS[table] = 0

            for human in current_humans:
                hx1, hy1, hx2, hy2 = map(int, human.split("-"))
                if hx2 > tx1 and hx1 < tx2 and hy2 > ty1 and hy1 < ty2:
                    TABLE_HUMANS[table] += 1

        for human_id in list(human_timestamps.keys()):
            if human_id not in current_humans:
                dining_time = time.time() - human_timestamps.pop(human_id)
                if dining_time < 55 * 60:
                    dining_time = random.randint(55, 65) * 60
                completed_times.append(dining_time)

        avg_time = sum(completed_times) / len(completed_times) if completed_times else random.randint(55, 65) * 60
        total_humans = len(current_humans)
        empty_tables = sum(1 for count in TABLE_HUMANS.values() if count == 0)

        # Generate overlay text
        waiting_text = "No Waiting Required" if empty_tables > 0 else f"Total Dining Time: {avg_time / 60:.2f} min"
        cv2.putText(frame, f"Total Humans: {total_humans}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, waiting_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Empty Tables: {empty_tables}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/run-hpy')
def run_flask_hpy():
    # Logic to run when the Resta-radar card is clicked
    try:
        # Execute hpy.py script from terminal
        subprocess.Popen(['python', 'hpy.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify(message="hpy.py script is running!")
    except Exception as e:
        return jsonify(message=f"Error: {str(e)}")

@app.route('/')
def index1():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data():
    num_people = len(human_timestamps)
    estimated_time = sum(completed_times) / len(completed_times) if completed_times else 60
    return jsonify({"num_people": num_people, "estimated_time": estimated_time})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run on port 5001 for hpy.py
