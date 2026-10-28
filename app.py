import os
import torch
import torch.nn as nn
import cv2
import pandas as pd
import numpy as np
import base64
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms, models
from collections import Counter

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

MODEL_PATH = 'best_efficientnet_gtsrb.pth'
LABEL_FILE = 'labels.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.60
IMAGE_SIZE = 224

class_df = pd.read_csv(LABEL_FILE)
class_names = class_df['Name'].tolist()

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 43)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    return img.unsqueeze(0).to(DEVICE)

def predict(tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
        return idx.item(), conf.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file"})
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file selected"})
    
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)
    
    try:
        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "Invalid image"})
        tensor = preprocess_image(img)
        idx, conf = predict(tensor)
        return jsonify({
            "class": class_names[idx],
            "confidence": round(conf * 100, 2),
            "confident": conf >= CONFIDENCE_THRESHOLD
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"})
        
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid frame"})
        
        tensor = preprocess_image(img)
        idx, conf = predict(tensor)
        return jsonify({
            "class": class_names[idx],
            "confidence": round(conf * 100, 2),
            "confident": conf >= CONFIDENCE_THRESHOLD
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file"})
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file selected"})
    
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)
    
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"})
        
        detections = []
        frame_count = 0
        max_frames = 300
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 3 == 0:  # every 3rd frame
                tensor = preprocess_image(frame)
                idx, conf = predict(tensor)
                if conf >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        "class": class_names[idx],
                        "confidence": round(conf * 100, 2),
                        "frame": frame_count
                    })
            frame_count += 1
        cap.release()
        
        if detections:
            classes = [d['class'] for d in detections]
            most_common = Counter(classes).most_common(1)[0]
            avg_conf = np.mean([d['confidence'] for d in detections if d['class'] == most_common[0]])
            return jsonify({
                "most_common_class": most_common[0],
                "count": most_common[1],
                "avg_confidence": round(avg_conf, 2),
                "total_detections": len(detections),
                "frames_processed": frame_count
            })
        return jsonify({
            "message": "No confident signs detected in video",
            "frames_processed": frame_count
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)