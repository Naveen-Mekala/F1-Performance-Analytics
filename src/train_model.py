import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("data/f1_results.csv")

# Keep only useful columns
df = df[
    [
        "Abbreviation",
        "TeamName",
        "Race",
        "GridPosition",
        "Points",
        "Position"
    ]
]

# Remove invalid positions
df = df[df["Position"].notna()]

# Create target
df["Podium"] = (df["Position"] <= 3).astype(int)

# Encode categorical columns
encoders = {}

for col in ["Abbreviation", "TeamName", "Race"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Features
X = df[
    [
        "Abbreviation",
        "TeamName",
        "Race",
        "GridPosition",
        "Points"
    ]
]

# Target
y = df["Podium"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "data/podium_model.pkl")

print("\nModel saved!")