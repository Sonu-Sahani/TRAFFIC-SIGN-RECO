import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ============================
#Load Trained PyTorch Model
# ============================
MODEL_PATH = 'traffic_sign_cnn.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model
model = TrafficSignCNN(num_classes=43).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ============================
#Image Preprocessing
# ============================
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Invalid image file. Unable to read the image.")
    
    img = cv2.resize(img, (32, 32))
    img = cv2.equalizeHist(img)  # Histogram equalization
    img = img.astype(np.float32) / 255.0  # Normalize and convert to float32

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img.to(torch.float32).to(DEVICE)  # Ensure float32 tensor

# ============================
#Get Class Name
# ============================
def getClassName(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[classNo] if 0 <= classNo < len(class_names) else "Unknown"

# ============================
#Model Prediction
# ============================
def model_predict(img_path):
    try:
        img = preprocess_image(img_path)
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.softmax(outputs, dim=1)
            classIndex = torch.argmax(probabilities).item()
            probabilityValue = torch.max(probabilities).item()

        return {
            "class": getClassName(classIndex),
            "confidence": round(probabilityValue * 100, 2)  # Convert to percentage
        }
    except Exception as e:
        return {"error": str(e)}

# ============================
#Flask Routes
# ============================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"})

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No file selected"})

    # Ensure uploads folder exists
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    # Save file securely
    file_path = os.path.join(upload_folder, secure_filename(f.filename))
    f.save(file_path)

    # Run prediction
    prediction = model_predict(file_path)
    return jsonify(prediction)  # Ensure JSON response


if __name__ == '__main__':
    app.run(port=5001, debug=True)
