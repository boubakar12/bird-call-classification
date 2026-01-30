import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
from tqdm import tqdm

SAMPLE_RATE = 22050
N_MFCC = 40
N_MELS = 40
TARGET_SEC = 3.0

def pad_or_trim(y, sr, target_sec=TARGET_SEC):
    target_len = int(sr * target_sec)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y

def stats_over_time(feat):
    return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

def extract_rich_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y = pad_or_trim(y, sr)


    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_d1 = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_con  = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness  = librosa.feature.spectral_flatness(y=y)

    # Time domain
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    feat_list = [
        stats_over_time(mfcc),
        stats_over_time(mfcc_d1),
        stats_over_time(mfcc_d2),
        stats_over_time(mel_db),
        stats_over_time(chroma),
        stats_over_time(spec_cent),
        stats_over_time(spec_bw),
        stats_over_time(spec_con),
        stats_over_time(spec_roll),
        stats_over_time(flatness),
        stats_over_time(zcr),
        stats_over_time(rms),
    ]
    return np.concatenate(feat_list)

def find_audio_path(folder, file_id, prefix="train"):
    candidates = [
        os.path.join(folder, f"{prefix}_{file_id}.wav"),
        os.path.join(folder, f"{file_id}.wav"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {file_id} in {folder}")

def build_feature_matrix(df, folder):
    X = []
    y = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["id"]
        label   = row["label"]
        path = find_audio_path(folder, file_id, prefix="train")
        feats = extract_rich_features(path)
        X.append(feats)
        y.append(label)
    return np.vstack(X), np.array(y, dtype=np.int64)


train_df_full = pd.read_csv("train.csv")
train_df_full = train_df_full.rename(columns={"# ID": "id", "Label": "label"})

# 80/20 split
train_df, val_df = train_test_split(
    train_df_full,
    test_size=0.2,
    random_state=42,
    stratify=train_df_full["label"]
)

print("Train size (rich+):", len(train_df))
print("Val size (rich+):", len(val_df))

X_train_rich, y_train_rich = build_feature_matrix(train_df, "train")
X_val_rich,   y_val_rich   = build_feature_matrix(val_df, "train")

print("X_train_rich shape:", X_train_rich.shape)
print("X_val_rich shape:", X_val_rich.shape)
np.save("X_train_rich.npy", X_train_rich)
np.save("y_train_rich.npy", y_train_rich)
np.save("X_val_rich.npy", X_val_rich)
np.save("y_val_rich.npy", y_val_rich)
print("Saved upgraded rich feature matrices.")
