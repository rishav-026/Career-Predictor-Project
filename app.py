from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import joblib
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score

app = Flask(__name__)
model_path = "improved_model_v2"
target_col = "What would you like to become when you grow up"

# Load trained model
predictor = TabularPredictor.load(model_path)
used_features = joblib.load(os.path.join(model_path, "used_features.pkl"))

@app.route('/')
def index():
    return render_template('index.html', dropped_rows=0)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    input_df = pd.read_csv(file)
    input_df.columns = input_df.columns.str.strip()

    # Feature engineering
    def tech_level(text):
        if pd.isnull(text):
            return "Unknown"
        elif "coding" in text.lower():
            return "High"
        elif "average" in text.lower():
            return "Medium"
        else:
            return "Low"

    input_df["Tech_Level"] = input_df["Tech-Savviness"].apply(tech_level)
    input_df["Academic Performance (CGPA/Percentage)"] = pd.to_numeric(
        input_df["Academic Performance (CGPA/Percentage)"], errors='coerce')
    input_df["Daily Water Intake (in Litres)"] = pd.to_numeric(
        input_df["Daily Water Intake (in Litres)"], errors='coerce')
    input_df["Number of Siblings"] = pd.to_numeric(
        input_df["Number of Siblings"], errors='coerce')

    input_df.drop(columns=["Timestamp", "Tech-Savviness"], inplace=True, errors="ignore")

    # Fill missing values
    risk_mapping = {
        "Low": 1,
        "Moderate": 2,
        "High": 3
    }
    if "Risk-Taking Ability" in input_df.columns:
        input_df["Risk-Taking Ability"] = input_df["Risk-Taking Ability"].map(risk_mapping)

    fill_values = {
        "Academic Performance (CGPA/Percentage)": input_df["Academic Performance (CGPA/Percentage)"].mean(),
        "Participation in Extracurricular Activities": "None",
        "Previous Work Experience (If Any)": "None",
        "Risk-Taking Ability": 2,
        "Leadership Experience": "None",
        "Networking & Social Skills": "Average",
        "Daily Water Intake (in Litres)": input_df["Daily Water Intake (in Litres)"].mean(),
        "Number of Siblings": input_df["Number of Siblings"].median(),
    }
    input_df.fillna(value=fill_values, inplace=True)

    # Drop rows missing critical features (should be 0 now)
    original_row_count = input_df.shape[0]
    missing_features = [col for col in used_features if col not in input_df.columns]
    if missing_features:
        return f"Missing features required by the model: {missing_features}", 400

    df_for_prediction = input_df.copy()
    prediction_input = df_for_prediction[used_features]
    dropped_rows = original_row_count - prediction_input.shape[0]

    if prediction_input.empty:
        return "No valid data to predict", 400

    # Make prediction
    predictions = predictor.predict(prediction_input)
    df_for_prediction["Predicted Career Goal"] = predictions

    # Accuracy check
    if target_col in df_for_prediction.columns:
        df_for_prediction[target_col] = df_for_prediction[target_col].astype(str).str.strip()
        df_for_prediction["Predicted Career Goal"] = df_for_prediction["Predicted Career Goal"].astype(str).str.strip()

        df_for_prediction["Correct"] = df_for_prediction[target_col] == df_for_prediction["Predicted Career Goal"]
        accuracy = accuracy_score(df_for_prediction[target_col], df_for_prediction["Predicted Career Goal"])
        accuracy_percent = round(accuracy * 100, 2)
    else:
        df_for_prediction["Correct"] = "N/A"
        accuracy_percent = "N/A"

    # Save results
    result_file = "static/predictions.csv"
    df_for_prediction.to_csv(result_file, index=False, encoding='utf-8-sig')

    # Build result table
    if target_col in df_for_prediction.columns:
        result_table = df_for_prediction[[target_col, "Predicted Career Goal", "Correct"]].to_html(classes='data', index=False)
    else:
        result_table = df_for_prediction[["Predicted Career Goal"]].to_html(classes='data', index=False)

    return render_template(
        "index.html",
        tables=result_table,
        accuracy=accuracy_percent,
        download_link="/download",
        dropped_rows=dropped_rows
    )

@app.route('/download')
def download():
    return send_file("static/predictions.csv", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
