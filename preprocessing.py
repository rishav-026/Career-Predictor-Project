import pandas as pd

def tech_level(text):
    if pd.isnull(text):
        return "Unknown"
    elif "coding" in text.lower():
        return "High"
    elif "average" in text.lower():
        return "Medium"
    else:
        return "Low"

def preprocess_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    if "Tech-Savviness" in df.columns:
        df["Tech_Level"] = df["Tech-Savviness"].apply(tech_level)

    df["Academic Performance (CGPA/Percentage)"] = pd.to_numeric(df.get("Academic Performance (CGPA/Percentage)"), errors='coerce')
    df["Daily Water Intake (in Litres)"] = pd.to_numeric(df.get("Daily Water Intake (in Litres)"), errors='coerce')
    df["Number of Siblings"] = pd.to_numeric(df.get("Number of Siblings"), errors='coerce')

    df.drop(columns=["Timestamp", "Tech-Savviness"], inplace=True, errors="ignore")

    if "Risk-Taking Ability" in df.columns:
        risk_mapping = {"Low": 1, "Moderate": 2, "High": 3}
        df["Risk-Taking Ability"] = df["Risk-Taking Ability"].map(risk_mapping)

    return df

def fill_missing_values(df):
    fill_values = {
        "Academic Performance (CGPA/Percentage)": df["Academic Performance (CGPA/Percentage)"].mean(),
        "Participation in Extracurricular Activities": "None",
        "Previous Work Experience (If Any)": "None",
        "Risk-Taking Ability": 2,
        "Leadership Experience": "None",
        "Networking & Social Skills": "Average",
        "Daily Water Intake (in Litres)": df["Daily Water Intake (in Litres)"].mean(),
        "Number of Siblings": df["Number of Siblings"].median(),
    }
    df.fillna(value=fill_values, inplace=True)
    return df
