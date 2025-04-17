import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import torch.nn as nn  # <-- Missing import added

# Load model
MODEL_PATH = 'traffic_sign_cnn.pth'
LABEL_FILE = 'labels.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
class_df = pd.read_csv(LABEL_FILE)
class_names = class_df['Name'].tolist()

# Model definition
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
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = TrafficSignCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_SIMPLEX

def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

while True:
    success, imgOriginal = cap.read()
    if not success:
        break
    
    img_gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (32, 32))
    img_eq = cv2.equalizeHist(img_resized)
    img = preprocess_image(img_eq)
    
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, 1)
        class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0][class_idx].item()
    
    cv2.putText(imgOriginal, f"CLASS: {class_names[class_idx]}", (20, 35), font, 0.75, (0, 0, 255), 2)
    cv2.putText(imgOriginal, f"CONFIDENCE: {confidence*100:.2f}%", (20, 75), font, 0.75, (0, 0, 255), 2)
    cv2.imshow("Traffic Sign Recognition", imgOriginal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()