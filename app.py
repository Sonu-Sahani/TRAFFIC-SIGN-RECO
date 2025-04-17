import os
import torch
import cv2
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and labels
MODEL_PATH = 'traffic_sign_cnn.pth'
LABEL_FILE = 'labels.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image")
    img = cv2.resize(img, (32, 32))
    img = cv2.equalizeHist(img)
    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        img = preprocess_image(filepath)
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.softmax(outputs, 1)
            class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[0][class_idx].item()
        
        return jsonify({
            "class": class_names[class_idx],
            "confidence": f"{confidence*100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)