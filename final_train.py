import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

SAMPLE_RATE = 22050
N_MFCC = 20

def extract_file(path):
    """Load wav file and extract MFCC features (mean + std)"""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def find_audio(folder, file_id, prefix):
    """Find the correct audio file path given folder, file_id, and prefix"""
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

def build_feature_matrix_train(df):
    """ X, y for ALL training samples"""

    X = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["id"]     
        label   = row["label"]
        path = find_audio("train", file_id, prefix="train")
        feats = extract_file(path)
        X.append(feats)
        y.append(label)
    return np.vstack(X), np.array(y, dtype=np.int64)

def build_feature_matrix_test(df):
    """ X for all test samples using"""
    X = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["# ID"]
        path = find_audio("test", file_id, prefix="test")
        feats = extract_file(path)
        X.append(feats)
    return np.vstack(X)


train_df_full = pd.read_csv("train.csv")
train_df_full = train_df_full.rename(columns={"# ID": "id", "Label": "label"})

sample_sub = pd.read_csv("sample_submit.csv")  

print("Number of train rows:", len(train_df_full))
print("Number of test rows:", len(sample_sub))

#  Extract
print("\nExtracting features for ALL training data...")
X_all, y_all = build_feature_matrix_train(train_df_full)

print("\nExtracting features for test data...")
X_test = build_feature_matrix_test(sample_sub)

print("\nX_all shape:", X_all.shape)
print("X_test shape:", X_test.shape)

# Train
rf = RandomForestClassifier(
    n_estimators=BEST_N,
    max_depth=BEST_DEPTH,
    min_samples_leaf=BEST_LEAF,
    max_features=BEST_MAX_FEATURES,
    random_state=42,
    n_jobs=-1
)
print("\nTraining final Random Forest on all data")
rf.fit(X_all, y_all)

# prediction
print("Predicting on test set...")
test_pred = rf.predict(X_test)

# Save submission
submission = sample_sub.copy()
submission["Label"] = test_pred    
submission.to_csv("my_submission.csv", index=False)
print("\nSaved my_submission_tune.csv")
