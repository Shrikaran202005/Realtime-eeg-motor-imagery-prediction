🧠 Real-Time EEG Motor Imagery Prediction System

Deep Learning + STFT + Streamlit-Based Real-Time BCI Interface

🚀 Overview

The Real-Time EEG Motor Imagery Prediction System is an end-to-end deep learning pipeline for classifying EEG motor imagery tasks using Short-Time Fourier Transform (STFT), a custom CNN architecture, and an interactive Streamlit interface.

The system converts raw EEG → spectrograms → CNN prediction → real-time visualization, enabling transparent and interpretable Brain–Computer Interface research.

📦 Installation Instructions

1️⃣ Clone the Repository
git clone https://github.com/Shrikaran202005/Realtime-eeg-motor-imagery-prediction.git

cd Real-Time-EEG-MI-Prediction-System

2️⃣ Create a Virtual Environment
python -m venv eegenv
source eegenv/bin/activate     # Linux/Mac
eegenv\Scripts\activate        # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt


If you don’t have requirements.txt, use:

pip install numpy scipy matplotlib streamlit torch torchvision torchaudio mne scikit-learn

4️⃣ Download EEG Dataset

Ensure you have your .gdf or converted .npy dataset placed inside:

/data/
    train/
    test/


(or update the path inside your script).

▶️ Usage Guide
🧪 1. Train the Model

Run the training script:

python train_model.py


This will:

Load training data

Perform STFT

Train EnhancedEEGCNNV3

Save the model as:

models/enhanced_eeg_cnn_v3.pth

🎛 2. Test the Model
python test_model.py

🎨 3. Launch the Streamlit App (Real-Time UI)
streamlit run app.py

The Streamlit UI allows you to:

Select trial index

Visualize EEG spectrogram

View class-wise prediction probabilities

View confusion matrix

Explore per-class accuracy

Monitor real-time predictions

📊 4. Real-Time Prediction Script (Optional)
python live_predict.py

🧠 Model Architecture Diagram (EnhancedEEGCNNV3)

Below is an ASCII-style conceptual architecture:

           ┌──────────────────────────────────────────────┐
           │           EnhancedEEGCNNV3 Model              │
           └──────────────────────────────────────────────┘
                           │
                           ▼
                ┌────────────────────┐
                │ Input Spectrogram  │  (120 × 32 × 3)
                └────────────────────┘
                           │
                           ▼
           ┌────────────────────────────────────────┐
           │   Conv2D → BatchNorm → ReLU → MaxPool  │
           └────────────────────────────────────────┘
                           │
                           ▼
           ┌────────────────────────────────────────┐
           │   Conv2D → BatchNorm → ReLU → MaxPool  │
           └────────────────────────────────────────┘
                           │
                           ▼
           ┌────────────────────────────────────────┐
           │        Dropout → Flatten Layer         │
           └────────────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │   Fully Connected  │
                  └────────────────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │   Softmax Output   │  (4 Classes)
                  └────────────────────┘


This architecture extracts spatial-temporal EEG features from spectrograms using deep convolutional layers and regularization techniques.


🧩 Features

✔ STFT-based spectral EEG preprocessing
✔ Deep CNN model (EnhancedEEGCNNV3)
✔ Real-time predictions using Streamlit
✔ Spectrogram visualization
✔ Confusion matrix and accuracy plots
✔ Modular and scalable BCI pipeline
✔ Fully interpretable prediction system
