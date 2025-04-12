import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.utils import resample
import os
import joblib
from sklearn.metrics import accuracy_score

# Load dataset
filename = "train_data1.csv"
df = pd.read_csv(filename)
df.columns = df.columns.str.strip()

# Set target column
target = "What would you like to become when you grow up"
df = df.dropna(subset=[target])

# Convert string columns (strip spaces)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# Feature Engineering: Tech Level
def tech_level(text):
    if pd.isnull(text):
        return "Unknown"
    elif "coding" in text.lower():
        return "High"
    elif "average" in text.lower():
        return "Medium"
    else:
        return "Low"

df["Tech_Level"] = df["Tech-Savviness"].apply(tech_level)

# Convert numeric fields safely
df["Academic Performance (CGPA/Percentage)"] = pd.to_numeric(df["Academic Performance (CGPA/Percentage)"], errors='coerce')
df["Daily Water Intake (in Litres)"] = pd.to_numeric(df["Daily Water Intake (in Litres)"], errors='coerce')
df["Number of Siblings"] = pd.to_numeric(df["Number of Siblings"], errors='coerce')

# Drop unused columns
df = df.drop(columns=["Timestamp", "Tech-Savviness"], errors="ignore")

# Drop rows with missing values
df = df.dropna()

# Check class balance
print("Original Class Distribution:\n", df[target].value_counts())

# Balance the dataset (undersampling)
grouped = [df[df[target] == label] for label in df[target].unique()]
min_count = min(len(g) for g in grouped)

df_balanced = pd.concat([
    resample(g, replace=False, n_samples=min_count, random_state=42)
    for g in grouped
])

print("\nBalanced Class Distribution:\n", df_balanced[target].value_counts())

# Train the model
predictor = TabularPredictor(
    label=target,
    problem_type="multiclass",
    path="improved_model_v2",
    eval_metric="accuracy"
).fit(
    train_data=df_balanced,
    time_limit=900,
    presets="best_quality"
)

# Evaluate the model
print("\nModel Leaderboard:")
print(predictor.leaderboard(silent=True))

# Predict on training data (optional sanity check)
df_to_predict = df_balanced.drop(columns=[target])

# Predict separately without modifying the original
predictions = predictor.predict(df_to_predict)
sample_df = df_balanced.copy()
sample_df["Predicted Career Goal"] = predictions

print("\nSample Predictions:")
print(sample_df[[target, "Predicted Career Goal"]].head(10))

acc = accuracy_score(sample_df[target], sample_df["Predicted Career Goal"])
print(f"\nTraining Accuracy: {acc * 100:.2f}%")

# Evaluate accuracy
print("\nEvaluation Metrics:")
predictor.evaluate(df_balanced)

print("\n✅ Training complete. Model saved to 'improved_model_v2/'")

# Save features used in training (before adding predictions)
used_features = [col for col in df_balanced.columns if col not in [target, "Predicted Career Goal"]]

os.makedirs("improved_model_v2", exist_ok=True)
joblib.dump(used_features, "improved_model_v2/used_features.pkl")

print("\n✅ Training complete. Model and features saved to 'improved_model_v2/'")
