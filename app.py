import os
import torch
import cv2
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for videos

# Load model and labels
MODEL_PATH = 'traffic_sign_cnn.pth'
LABEL_FILE = 'labels.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.65  # Only show strong predictions

class_df = pd.read_csv(LABEL_FILE)
class_names = class_df['Name'].tolist()

class TrafficSignCNN(torch.nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = TrafficSignCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def preprocess_image(img):
    """img: numpy array (BGR or grayscale)"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    img = cv2.equalizeHist(img)
    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

def predict_from_tensor(tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, 1)
        class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0][class_idx].item()
    return class_idx, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"error": "Invalid image"})
        tensor = preprocess_image(img)
        class_idx, confidence = predict_from_tensor(tensor)
        
        result = {
            "class": class_names[class_idx],
            "confidence": round(confidence * 100, 2),
            "confident": confidence >= CONFIDENCE_THRESHOLD
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """For live camera - receives base64 image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"})
        
        # Decode base64
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid frame"})
        
        tensor = preprocess_image(img)
        class_idx, confidence = predict_from_tensor(tensor)
        
        result = {
            "class": class_names[class_idx],
            "confidence": round(confidence * 100, 2),
            "confident": confidence >= CONFIDENCE_THRESHOLD
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Process uploaded video and return summary of detections"""
    if 'file' not in request.files:
        return jsonify({"error": "No file"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        cap = cv2.VideoCapture(filepath)
        detections = []
        frame_count = 0
        max_frames = 150  # Limit for performance
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for speed
            if frame_count % 5 == 0:
                tensor = preprocess_image(frame)
                class_idx, confidence = predict_from_tensor(tensor)
                if confidence >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        "class": class_names[class_idx],
                        "confidence": round(confidence * 100, 2),
                        "frame": frame_count
                    })
            frame_count += 1
        
        cap.release()
        
        # Get most common prediction
        if detections:
            from collections import Counter
            classes = [d['class'] for d in detections]
            most_common = Counter(classes).most_common(1)[0]
            avg_conf = np.mean([d['confidence'] for d in detections if d['class'] == most_common[0]])
            summary = {
                "most_common_class": most_common[0],
                "count": most_common[1],
                "avg_confidence": round(avg_conf, 2),
                "total_detections": len(detections),
                "frames_processed": frame_count
            }
        else:
            summary = {"message": "No confident detections found", "frames_processed": frame_count}
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)