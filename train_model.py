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

# Strip whitespace from all string columns
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

# Balance the dataset (undersampling)
grouped = [df[df[target] == label] for label in df[target].unique()]
min_count = min(len(g) for g in grouped)

df_balanced = pd.concat([
    resample(g, replace=False, n_samples=min_count, random_state=42)
    for g in grouped
])

# Save fill values from balanced clean data
fill_values = {
    "Academic Performance (CGPA/Percentage)": df_balanced["Academic Performance (CGPA/Percentage)"].mean(),
    "Participation in Extracurricular Activities": "None",
    "Previous Work Experience (If Any)": "None",
    "Risk-Taking Ability": 2,
    "Leadership Experience": "None",
    "Networking & Social Skills": "Average",
    "Daily Water Intake (in Litres)": df_balanced["Daily Water Intake (in Litres)"].mean(),
    "Number of Siblings": df_balanced["Number of Siblings"].median(),
}

os.makedirs("improved_model_v2", exist_ok=True)
joblib.dump(fill_values, "improved_model_v2/fill_values.pkl")

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

# Predict on training data for sanity check
df_to_predict = df_balanced.drop(columns=[target])
predictions = predictor.predict(df_to_predict)

sample_df = df_balanced.copy()
sample_df["Predicted Career Goal"] = predictions
acc = accuracy_score(sample_df[target], sample_df["Predicted Career Goal"])
print(f"\nTraining Accuracy: {acc * 100:.2f}%")

# Save used features
used_features = [col for col in df_balanced.columns if col not in [target, "Predicted Career Goal"]]
joblib.dump(used_features, "improved_model_v2/used_features.pkl")

print("\n✅ Training complete. Model and artifacts saved to 'improved_model_v2/'")
