import pandas as pd

train_df = pd.read_csv("train.csv")
sample_sub = pd.read_csv("sample_submit.csv")

# Original colums
print("Original columns:", train_df.columns)
print("sample_submit columns:", sample_sub.columns)

# Rename
train_df = train_df.rename(columns={"# ID": "id", "Label": "label"})
sample_sub = sample_sub.rename(columns={"# ID": "id", "Label": "label"})

print("\nTrain.csv head (renamed):")
print(train_df.head())
print("\nLabel counts:")
print(train_df["label"].value_counts())
print("\nSample submission head (renamed):")
print(sample_sub.head())
