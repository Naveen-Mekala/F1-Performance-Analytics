import pandas as pd

df = pd.read_csv("data/f1_results.csv")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())