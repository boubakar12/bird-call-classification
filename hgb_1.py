import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier

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

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_con  = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness  = librosa.feature.spectral_flatness(y=y)

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

def find_audio_path(folder, file_id, prefix):
    candidates = [
        os.path.join(folder, f"{prefix}_{file_id}.wav"),
        os.path.join(folder, f"{file_id}.wav"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find audio file for id={file_id} in {folder}. "
        f"Tried: {', '.join(candidates)}"
    )

def build_feature_matrix_all_test(df):
    X = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["# ID"]
        path = find_audio_path("test", file_id, prefix="test")
        feats = extract_rich_features(path)
        X.append(feats)
    return np.vstack(X)

# Load cached
X_train_rich = np.load("X_train_rich.npy")
y_train_rich = np.load("y_train_rich.npy")
X_val_rich   = np.load("X_val_rich.npy")
y_val_rich   = np.load("y_val_rich.npy")

X_all_rich = np.vstack([X_train_rich, X_val_rich])
y_all      = np.concatenate([y_train_rich, y_val_rich])

print("X_all_rich shape:", X_all_rich.shape)
print("y_all shape:", y_all.shape)

# Load test ID
sample_sub = pd.read_csv("sample_submit.csv")
print("Number of test rows:", len(sample_sub))

# Extract test features
print("\nExtracting rich+ features for ALL test data...")
X_test_rich = build_feature_matrix_all_test(sample_sub)
print("X_test_rich shape:", X_test_rich.shape)

BEST_HGB_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 10,
    "max_leaf_nodes": 63,
    "max_iter": 300,
}

n_classes = len(np.unique(y_all))
sum_proba = np.zeros((X_test_rich.shape[0], n_classes), dtype=float)
n_models = 0

# Train HGB models
N_BAGS = 5
for bag in range(N_BAGS):
    print(f"\n=== Training HGB bag {bag+1}/{N_BAGS} ===")
    rng = np.random.RandomState(42 + bag)
    idx = rng.choice(len(X_all_rich), size=int(0.9 * len(X_all_rich)), replace=False)
    X_sub = X_all_rich[idx]
    y_sub = y_all[idx]

    hgb = HistGradientBoostingClassifier(
        learning_rate=BEST_HGB_PARAMS["learning_rate"],
        max_depth=BEST_HGB_PARAMS["max_depth"],
        max_leaf_nodes=BEST_HGB_PARAMS["max_leaf_nodes"],
        max_iter=BEST_HGB_PARAMS["max_iter"],
        random_state=100 + bag
    )

    hgb.fit(X_sub, y_sub)
    proba = hgb.predict_proba(X_test_rich)
    sum_proba += proba
    n_models += 1

print(f"\nTotal HGB models in ensemble: {n_models}")
avg_proba = sum_proba / n_models

test_pred = np.argmax(avg_proba, axis=1) + 1  

submission = sample_sub.copy()
submission["Label"] = test_pred
submission.to_csv("my_submission_final.csv", index=False)
print("\nSaved my_submission_final.csv")
