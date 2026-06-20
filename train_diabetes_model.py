print("Starting script...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Libraries imported...")

# Load dataset
df = pd.read_csv("datasets/diabetes.csv")

print("Dataset loaded successfully!")
print(df.head())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("Preparing train-test split...")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

print("Model trained!")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "models/diabetes_model.pkl")

print("Model saved successfully!")