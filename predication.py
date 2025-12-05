from model import EnhancedEEGCNNV3 
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(page_title="EEG Motor Imagery Predictor", layout="wide")

# -----------------------------
# Load Data & Model
# -----------------------------
@st.cache_data
def load_data():
    X = np.load("X_preprocessed_2a.npy")  # (48888, 120, 32, 3)
    y = np.load("y_preprocessed_2a.npy")  # (48888,)
    return X, y

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedEEGCNNV3(num_classes=4).to(device)
    model.load_state_dict(torch.load("cnn_eeg_model.pth", map_location=device))
    model.eval()
    return model, device

X, y = load_data()
model, device = load_model()
class_map = {0: "Left Hand", 1: "Right Hand", 2: "Feet", 3: "Tongue"}

# -----------------------------
# Sidebar: Select trial
# -----------------------------
st.sidebar.title("EEG Motor Imagery Predictor")
trial_idx = st.sidebar.number_input(
    "Select Trial Index", min_value=0, max_value=len(X)-1, value=0, step=1
)

# -----------------------------
# Prepare Trial Input
# -----------------------------
X_trial = X[trial_idx]
y_true = y[trial_idx]

if X_trial.shape[-1] == 3:
    X_trial_t = np.transpose(X_trial, (2, 0, 1))  # (3,120,32)
else:
    X_trial_t = X_trial

X_tensor = torch.tensor(X_trial_t, dtype=torch.float32).unsqueeze(0).to(device)

# -----------------------------
# Predict
# -----------------------------
with torch.no_grad():
    logits = model(X_tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    pred_class = np.argmax(probs)

# -----------------------------
# Show Input EEG
# -----------------------------
st.subheader("Input EEG Spectrogram")
fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(X_trial_t[0], aspect="auto", cmap="viridis")
ax.set_xlabel("Time")
ax.set_ylabel("Frequency bins")
ax.set_title(f"True Class: {class_map[y_true]}")
st.pyplot(fig)

# -----------------------------
# Show Prediction Probabilities
# -----------------------------
st.subheader("Prediction Probabilities")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(class_map.values(), probs, color="skyblue")
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title(f"Predicted: {class_map[pred_class]}")
for i, v in enumerate(probs):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
st.pyplot(fig)

# -----------------------------
# Dataset Overview (optional)
# -----------------------------
if st.sidebar.checkbox("Show Dataset Evaluation"):
    batch_size = 16
    if X.shape[-1] == 3:
        X_eval = np.transpose(X, (0, 3, 1, 2))
    else:
        X_eval = X
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_eval), batch_size):
            X_batch = torch.tensor(X_eval[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(X_batch)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(batch_preds)
    preds = np.concatenate(preds)
    overall_acc = accuracy_score(y, preds)
    st.write(f"**Overall Dataset Accuracy:** {overall_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_map.values(), yticklabels=class_map.values(), cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Per-class accuracy
    per_class_acc = cm.diagonal()/cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(class_map.values(), per_class_acc, color="skyblue")
    ax.set_ylim(0,1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    for i, v in enumerate(per_class_acc):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    st.pyplot(fig)
