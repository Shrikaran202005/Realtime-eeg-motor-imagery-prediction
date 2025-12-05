import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from preprocess import preprocess_single_file
from model import EnhancedEEGCNNV3

# ----------------------------
# Paths and device
# ----------------------------
DATA_FOLDER = "data/GDF/2a_train/"
MODEL_PATH = "cnn_eeg_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. Load Model
# ----------------------------
model = EnhancedEEGCNNV3(num_classes=4).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ Model loaded successfully")

# ----------------------------
# 2. Set PyTorch deterministic mode
# ----------------------------
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# 3. List all GDF files
# ----------------------------
file_list = glob.glob(DATA_FOLDER + "*.gdf")

# ----------------------------
# 4. Loop over files
# ----------------------------
class_names = ["Left Hand", "Right Hand", "Foot", "Tongue"]

for file_path in file_list:
    print(f"\nProcessing file: {file_path}")

    try:
        # Preprocess EEG file
        X, _ = preprocess_single_file(file_path)
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)
        
        # Convert to torch format (N, C, H, W)
        X_torch = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32).to(DEVICE)

        # Window-level predictions
        with torch.no_grad():
            outputs = model(X_torch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        pred_labels = [class_names[i] for i in preds]

        # Display predictions
        print("Window-level Predictions:")
        for i, label in enumerate(pred_labels):
            print(f"Window {i+1:02d} → Predicted: {label}")

        # Optional: plot
        plt.figure(figsize=(12,5))
        plt.plot(pred_labels, marker='o')
        plt.title(f"Window-level Predictions for {file_path.split('/')[-1]}")
        plt.xlabel("Window Index")
        plt.ylabel("Predicted Class")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"⚠️ Skipping file {file_path} due to error: {e}")
