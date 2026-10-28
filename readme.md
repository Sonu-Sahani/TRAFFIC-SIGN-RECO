# Traffic Sign Recognition

EfficientNet-B0 model trained on full GTSRB dataset.

## Features
- Image Upload prediction
- Live Camera detection
- Video Upload analysis

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open: http://localhost:5001

## Model
`best_efficientnet_gtsrb.pth` (EfficientNet-B0)

## Note
Model works best on GTSRB-style images. Real-world photos may have lower accuracy due to domain shift.
