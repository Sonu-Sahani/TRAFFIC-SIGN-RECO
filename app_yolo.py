import os
import cv2
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

# Class names (short)
class_names = ['20','30','50','60','70','80','end80','100','120','no_pass','no_pass_3.5','right_of_way','priority','yield','stop','no_veh','no_3.5','no_entry','caution','curve_left','curve_right','double_curve','bumpy','slippery','narrow','work','signals','pedestrian','children','bicycle','ice','animals','end_limits','turn_right','turn_left','ahead','straight_right','straight_left','keep_right','keep_left','roundabout','end_no_pass','end_no_pass_3.5']

def predict_image(img):
    results = model(img, conf=0.4, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "class": class_names[cls] if cls < len(class_names) else str(cls),
                "confidence": round(conf * 100, 2)
            })
    return detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file"})
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file"})
    
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)
    
    try:
        img = cv2.imread(path)
        dets = predict_image(img)
        if dets:
            best = max(dets, key=lambda x: x['confidence'])
            return jsonify({"class": best['class'], "confidence": best['confidence'], "confident": best['confidence'] > 50})
        return jsonify({"class": "No sign detected", "confidence": 0, "confident": False})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.get_json()
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        dets = predict_image(img)
        if dets:
            best = max(dets, key=lambda x: x['confidence'])
            return jsonify({"class": best['class'], "confidence": best['confidence'], "confident": best['confidence'] > 50})
        return jsonify({"class": "No sign", "confidence": 0, "confident": False})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file"})
    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)
    
    try:
        cap = cv2.VideoCapture(path)
        detections = []
        frame_count = 0
        while cap.isOpened() and frame_count < 150:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % 5 == 0:
                dets = predict_image(frame)
                for d in dets:
                    if d['confidence'] > 50:
                        detections.append(d['class'])
            frame_count += 1
        cap.release()
        
        if detections:
            most = Counter(detections).most_common(1)[0]
            return jsonify({"most_common_class": most[0], "count": most[1], "total_detections": len(detections)})
        return jsonify({"message": "No signs detected"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)