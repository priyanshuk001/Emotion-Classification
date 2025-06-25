Here's a revised and personalized version of your README to reflect your ownership, coding style, and deployment context more clearly, while preserving technical accuracy and structure.

---

# 🗣️ Speech Emotion Recognition using Deep Learning

## 📌 Objective

This project aims to build an end-to-end **speech emotion classification pipeline** that leverages deep learning and audio signal processing to detect emotions from human voice recordings. The solution supports both **speech and song-based input**, using robust feature extraction and a CNN-based classifier.

---

## 🚀 What This Project Does

* Extracts key audio features: MFCC, ZCR, RMSE
* Performs audio augmentation for better generalization
* Trains a custom 1D CNN for emotion classification
* Deploys an interactive **Streamlit web app** for:

  * Real-time single/batch predictions
  * Audio playback
  * Confidence visualization (pie/bar)
  * Downloadable CSV results

---

## 🗃️ Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
🔗 [Download here](https://zenodo.org/records/1188976#.XCx-tc9KhQI)

* 🎧 Total Samples: 9808 audio clips
* Emotions covered:

  * Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

| Emotion   | Count |
| --------- | ----- |
| Calm      | 376   |
| Happy     | 376   |
| Sad       | 376   |
| Angry     | 376   |
| Fearful   | 376   |
| Disgust   | 192   |
| Surprised | 192   |
| Neutral   | 188   |

---

## 🔬 Methodology

### 🎧 Audio Preprocessing

* Sampling Rate: 22050 Hz
* Frame Length: 2048, Hop Length: 512
* Extracted Features:

  * MFCC (13 Coefficients)
  * Zero Crossing Rate
  * Root Mean Square Energy
* Combined Feature Vector Size: **2376**

### 🧪 Data Augmentation

* Time Stretching
* Pitch Shifting
* White Noise Addition
* Audio Shifting
* Speed Change

### 🧠 Model Architecture

Custom 1D CNN:

* 5 Conv1D + BatchNorm + MaxPool layers
* Dropout layers for regularization
* Dense layer (512 units)
* Softmax output (8 emotion classes)
  📊 **Trainable Params**: \~7.19 Million

### ⚙️ Training Setup

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Batch Size: 64, Epochs: 40
* Learning Rate Scheduler: `ReduceLROnPlateau`

---

## 📈 Results

* ✅ **Test Accuracy:** 93.93%
* 📉 **F1 Scores:**

  * Macro F1: 93.86%
  * Weighted F1: 93.94%
  * Micro F1: 93.93%

---

## 🌐 Streamlit Web Application

An interactive web app allows users to:

* 📤 Upload single or multiple `.wav` files
* 🎧 Play audio interactively
* 📊 See predictions + confidence scores (bar/pie charts)
* 📈 View waveform and histograms
* ⬇️ Export batch results as CSV

### Live Features

* 🔄 Auto model loading
* ⚡ Real-time inference
* 📊 Visual summaries of emotion distribution

---

## 📁 Code Structure

```
📦 Emotion_Classifier_App/
├── app.py                  # Streamlit Web App
├── model/
│   ├── cnn_model_full.keras #you can downlaod the model from drive link in models.md folder
│   ├── scaler.pkl
│   └── encoder.pkl
├── utils.py                # Feature extraction + helper functions
├── notebooks/
│   └── training_pipeline.ipynb
├── requirements.txt
└── assets/                 # Optional sample audio files
```

---

## 🧪 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/emotion-classifier-app.git
cd emotion-classifier-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate  # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

* **Deep Learning**: TensorFlow, Keras
* **Audio**: Librosa
* **Visualization**: Plotly, Matplotlib, Seaborn
* **Web Interface**: Streamlit
* **Utilities**: Scikit-learn, NumPy, Pandas

---

## 🙏 Acknowledgements

* [RAVDESS Dataset](https://zenodo.org/record/1188976) by Ryerson University
* TensorFlow, Librosa, and the open-source ML community

---

