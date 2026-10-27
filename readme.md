# AI-Based Traffic Sign Recognition

An AI-based traffic sign recognition system built using **PyTorch and CNN**. Classifies traffic signs into **43 classes** with support for image upload, **live camera**, and **video analysis**.

## Features

* Classifies **43 traffic sign categories**
* Custom CNN with PyTorch
* **Image Upload** prediction via Flask
* **Live Camera** real-time detection (browser)
* **Video Upload** analysis with summary of detections
* Confidence threshold (65%+) for better accuracy display
* GPU support when available

## Tech Stack

* Python, PyTorch, OpenCV, Flask, NumPy, Pandas

## Quick Start

```bash
git clone https://github.com/Sonu-Sahani/TRAFFIC-SIGN-RECO.git
cd TRAFFIC-SIGN-RECO
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:5001`

### Modes
1. **Image Upload** — Upload a photo and get prediction + confidence
2. **Live Camera** — Allow camera access for real-time detection
3. **Video Upload** — Upload a video and get the most detected sign summary

## Model

Trained CNN saved as `traffic_sign_cnn.pth`. Retrain with:
```bash
python main.py
```

## Author

**Sonu Sahani** — [GitHub](https://github.com/Sonu-Sahani)
