import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/single_genre_artists.csv")

print("\nDataset Shape")
print(df.shape)

print("\nFirst 5 Rows")
print(df.head())

print("\nColumn Names")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\n Summary")
print(df.describe())