# AI-Based Traffic Sign Recognition

An AI-based traffic sign recognition system built using **PyTorch and Convolutional Neural Networks (CNN)**. The system can classify traffic signs into **43 different classes** and supports both **image-based prediction** through a Flask web interface and **real-time recognition** using a webcam.

## Features

* Classifies **43 different traffic sign categories**
* Custom CNN architecture built with PyTorch
* Image preprocessing using:

  * Grayscale conversion
  * Image resizing to `32 × 32`
  * Histogram equalization
  * Normalization
* Image-based traffic sign prediction through a Flask web application
* Real-time traffic sign recognition using OpenCV webcam input
* Displays predicted class and confidence score
* GPU support when CUDA is available

## Tech Stack

* **Python**
* **PyTorch**
* **OpenCV**
* **Flask**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Torchvision**

## Model Architecture

The project uses a custom CNN architecture:

```text
Input Image (1 × 32 × 32)
        ↓
Convolutional Layer (1 → 32)
        ↓
ReLU + Max Pooling
        ↓
Convolutional Layer (32 → 64)
        ↓
ReLU + Max Pooling
        ↓
Convolutional Layer (64 → 128)
        ↓
ReLU + Max Pooling
        ↓
Flatten
        ↓
Fully Connected Layer (2048 → 512)
        ↓
ReLU + Dropout
        ↓
Fully Connected Layer (512 → 43)
        ↓
Predicted Traffic Sign
```

## Dataset

The dataset is organized into **43 class folders**, where each folder represents a traffic sign category.

```text
Dataset/
├── 0/
├── 1/
├── 2/
├── ...
├── 41/
└── 42/
```

The class labels are stored in `labels.csv`.

## Data Preprocessing

Each input image goes through the following preprocessing pipeline:

1. Convert image to grayscale.
2. Resize image to `32 × 32`.
3. Apply histogram equalization using OpenCV.
4. Convert the image into a PyTorch tensor.
5. Normalize pixel values.

The same preprocessing pipeline is used during both training and prediction.

## Training

The dataset is divided into:

* **60% Training**
* **20% Validation**
* **20% Testing**

### Training Configuration

| Parameter      |              Value |
| -------------- | -----------------: |
| Batch Size     |                 32 |
| Epochs         |                 10 |
| Learning Rate  |              0.001 |
| Optimizer      |               Adam |
| Loss Function  | Cross Entropy Loss |
| Input Size     |            32 × 32 |
| Output Classes |                 43 |

The trained model is saved as:

```text
traffic_sign_cnn.pth
```

## Project Structure

```text
TRAFFIC-SIGN-RECO/
│
├── Dataset/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 42/
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   ├── base.html
│   └── index.html
│
├── uploads/
│
├── app.py
├── main.py
├── test.py
├── labels.csv
└── traffic_sign_cnn.pth
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Sonu-Sahani/TRAFFIC-SIGN-RECO.git
cd TRAFFIC-SIGN-RECO
```

Install the required dependencies:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib flask
```

## Train the Model

To train the CNN model from scratch:

```bash
python main.py
```

The script will:

1. Load the dataset.
2. Preprocess the images.
3. Split the dataset into training, validation, and testing sets.
4. Train the CNN.
5. Display training and validation loss.
6. Evaluate the model on the test dataset.
7. Save the trained model as `traffic_sign_cnn.pth`.

## Image Prediction using Flask

Start the Flask application:

```bash
python app.py
```

The application runs on:

```text
http://localhost:5001
```

Upload a traffic sign image through the web interface. The application processes the image and returns:

```text
Predicted Class
Confidence Score
```

## Real-Time Webcam Recognition

To run real-time traffic sign recognition:

```bash
python test.py
```

The application opens the webcam and continuously:

1. Captures the current frame.
2. Converts it to grayscale.
3. Resizes it to `32 × 32`.
4. Applies histogram equalization.
5. Passes the image through the trained CNN.
6. Calculates class probabilities using Softmax.
7. Displays the predicted traffic sign and confidence score.

Press **Q** to exit the webcam window.

## Prediction Pipeline

```text
Image / Webcam Frame
        ↓
Grayscale Conversion
        ↓
Resize to 32 × 32
        ↓
Histogram Equalization
        ↓
Normalization
        ↓
CNN Model
        ↓
Softmax
        ↓
Predicted Class + Confidence
```

## Example Output

The system displays the prediction in the following format:

```text
CLASS: Stop
CONFIDENCE: 96.42%
```

## Model Output

The final fully connected layer produces **43 output values**, one for each traffic sign class.

Softmax is applied to convert these outputs into probabilities:

```text
CNN Output
    ↓
Softmax
    ↓
Class Probabilities
    ↓
Highest Probability
    ↓
Predicted Traffic Sign
```

## Future Improvements

* Use data augmentation such as rotation, cropping, and brightness variation.
* Improve performance using Batch Normalization.
* Experiment with deeper CNN architectures.
* Add top-K predictions.
* Add confidence thresholding to handle uncertain predictions.
* Deploy the Flask application to a cloud platform.
* Optimize the model for real-time edge devices.

## Author

**Sonu Sahani**

GitHub: [Sonu-Sahani](https://github.com/Sonu-Sahani)

## License

This project is created for educational and learning purposes.
