# Traffic Sign Recognition (Modern Version)

High-accuracy traffic sign classification using **ResNet18** (pretrained) + strong data augmentation. Supports **Image Upload**, **Live Camera**, and **Video Analysis**.

## Why Better?
- Pretrained ResNet18 → much better real-world accuracy
- Heavy data augmentation (rotation, color, brightness, scale)
- RGB images instead of only grayscale
- Confidence threshold for reliable predictions

## Quick Start

```bash
git clone https://github.com/Sonu-Sahani/TRAFFIC-SIGN-RECO.git
cd TRAFFIC-SIGN-RECO
pip install -r requirements.txt
```

### 1. Train the model (one time)
```bash
python train.py
```
This creates `best_traffic_sign_resnet.pth`.  
**Recommended:** Use GPU. CPU pe 1–2 hours lag sakte hain.

### 2. Run the web app
```bash
python app.py
```
Open: http://localhost:5001

## Features
- Image Upload prediction
- Live Camera real-time detection
- Video Upload → most common sign summary
- High confidence badge (only shows strong predictions)

## Tech Stack
Python • PyTorch • ResNet18 • OpenCV • Flask • Bootstrap

## Author
**Sonu Sahani** — [GitHub](https://github.com/Sonu-Sahani)
