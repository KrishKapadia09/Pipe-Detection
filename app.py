import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os
import socket

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO(r'C:\Users\use\OneDrive\Desktop\Image Processing\Pipe detection\runs\detect\train\weights\best.pt')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
VIDEO_FOLDER = 'videos'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)

recording = False
out = None

@app.route('/')
def index():
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template('index_mobile.html')
    else:
        return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        frame = cv2.imread(file_path)
        results = model(frame)

        object_count = {}
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label != 'abel':  # Skip 'abel' label
                        if label not in object_count:
                            object_count[label] = 1
                        else:
                            object_count[label] += 1
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)  # Label the box

        count_text = ' '.join([f"{k}: {v}" for k, v in object_count.items()])
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        output_filename = 'annotated_' + file.filename
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        cv2.imwrite(output_path, frame)

        total_count = sum(object_count.values())
        return redirect(url_for('display_image', filename=output_filename, total_count=total_count))

@app.route('/display/<filename>')
def display_image(filename):
    total_count = request.args.get('total_count')
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template('display_mobile.html', filename=filename, total_count=total_count)
    else:
        return render_template('display.html', filename=filename, total_count=total_count)

@app.route('/results/<filename>')
def send_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

def generate_frames():
    global recording, out

    cap = cv2.VideoCapture(0)  # Use default camera

    if recording:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(VIDEO_FOLDER, 'live_feed.avi'), fourcc, 20.0, (640, 480))

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLO detection
        results = model(frame)
        object_count = {}
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label != 'abel':  # Skip 'abel' label
                        if label not in object_count:
                            object_count[label] = 1
                        else:
                            object_count[label] += 1

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # Define points for the polygon
                        points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                        points = points.reshape((-1, 1, 2))

                        # Draw the polygon
                        cv2.polylines(frame, [points], isClosed=True, color=(255, 255, 0), thickness=2)
                        # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)  # Label the box

        count_text = ' '.join([f"{k}: {v}" for k, v in object_count.items()])
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if recording:
            out.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    if recording:
        out.release()

@app.route('/live')
def live():
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template('mobile_live.html')
    else:
        return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    frame = np.array(image)

    results = model(frame)

    object_count = 0
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label != 'abel':  # Skip 'abel' label
                    object_count += 1

    return jsonify({'count': object_count})

@app.route('/live_count')
def live_count():
    return jsonify({'count': global_count})

@app.route('/stop_live', methods=['POST'])
def stop_live():
    global recording
    recording = False
    return jsonify({'message': 'Live feed stopped and video saved.'})

@app.route('/start_live', methods=['POST'])
def start_live():
    global recording
    recording = True
    return jsonify({'message': 'Live feed started and video recording.'})

if __name__ == "__main__":
    context = ('cert.pem', 'key.pem')  # Use the generated certificate and key
    app.run(debug=True, host='0.0.0.0', ssl_context=context)
