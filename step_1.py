# step_1.py
import pandas as pd

train_df = pd.read_csv("train.csv")
sample_sub = pd.read_csv("sample_submit.csv")

print("Train columns:", train_df.columns)
print("Sample_submit columns:", sample_sub.columns)

# Rename to nicer names (not strictly needed later, but helpful)
train_df = train_df.rename(columns={"# ID": "id", "Label": "label"})
sample_sub = sample_sub.rename(columns={"# ID": "id", "Label": "label"})

print("\nTrain head (renamed):")
print(train_df.head())

print("\nLabel counts:")
print(train_df["label"].value_counts())

print("\nSample submission head (renamed):")
print(sample_sub.head())
