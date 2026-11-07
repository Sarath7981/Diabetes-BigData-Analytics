from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

print("Downloading UCI Diabetes Health Indicators dataset...")

# Fetch dataset
dataset = fetch_ucirepo(id=891)

# Combine features and target
X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

# Create output directory
os.makedirs("data/raw", exist_ok=True)

# Save dataset
output_path = "data/raw/diabetes.csv"
df.to_csv(output_path, index=False)

print("\nDataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nTarget distribution:")
print(df["Diabetes_binary"].value_counts())
