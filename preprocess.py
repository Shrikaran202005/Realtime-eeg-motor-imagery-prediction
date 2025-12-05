# preprocess.py
import mne
import numpy as np
from scipy.signal import stft

# Parameters
TARGET_CHANNELS = ["EEG-C3", "EEG-Cz", "EEG-C4"]   # GDF names (check dataset)
SFREQ = 250  # BCI Competition IV-2a sampling rate
TMIN, TMAX = 0, 4  # trial duration in seconds
N_PER_SEG = 64
N_OVERLAP = 32

def preprocess_single_file(file_path):
    """Preprocess a single GDF file -> (X, y)."""
    # Load raw EEG
    raw = mne.io.read_raw_gdf(file_path, preload=True)

    # Clean up channel names
    raw.rename_channels(lambda x: x.strip())

    # Remove duplicate channels
    _, unique_idx = np.unique(raw.ch_names, return_index=True)
    raw.pick_channels([raw.ch_names[i] for i in sorted(unique_idx)])

    # Keep only C3, Cz, C4 (if they exist after duplicate removal)
    available_channels = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available_channels)

    # Extract events
    events, _ = mne.events_from_annotations(raw)
    print("Available events in file:")
    print(np.unique(events[:, 2]))

    # Four motor imagery classes
    event_id = {
        "left_hand": 1,   # replace with correct ID
        "right_hand": 2,
        "foot": 3,
        "tongue": 4,
    }

    # Epoching (0–4s windows)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True
    )

    X = []
    y = []

    for epoch, label in zip(epochs.get_data(), epochs.events[:, -1]):
        # epoch shape: (channels, time_points)
        spectrograms = []

        for ch in range(len(available_channels)):
            f, t, Zxx = stft(
                epoch[ch], fs=SFREQ, nperseg=N_PER_SEG, noverlap=N_OVERLAP
            )

            # Keep mu (8–12Hz) and beta (18–26Hz) bands
            freq_mask = (f >= 8) & (f <= 26)
            Zxx = np.abs(Zxx[freq_mask, :])

            # Normalize
            Zxx = (Zxx - Zxx.mean()) / (Zxx.std() + 1e-6)

            spectrograms.append(Zxx)

        # Stack → (freqs, times, channels)
        spectrograms = np.stack(spectrograms, axis=-1)
        X.append(spectrograms)

        # Map labels: 769→0, 770→1, 771→2, 772→3
        if label == 769:
            y.append(0)
        elif label == 770:
            y.append(1)
        elif label == 771:
            y.append(2)
        elif label == 772:
            y.append(3)

    X = np.array(X)  # (N, H, W, channels)
    y = np.array(y)

    print(f"Preprocessed {len(X)} trials → X {X.shape}, y {y.shape}")

    return X, y
