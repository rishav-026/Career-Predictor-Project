# 🎓 Career Prediction App

A full-stack machine learning application that predicts a student's likely future career path based on academic, behavioral, and personal traits. This project was built in just **8 hours** during the **HackMarch Hackathon** at **KLE Society**.

It features a **Python backend** with a trained ML model using **AutoGluon**, and a simple frontend built with HTML, CSS, and JavaScript.

---

## 🚀 Project Highlights

- 🔍 Predicts career goals from personal and academic inputs
- 🧠 AutoML-based model using AutoGluon (`TabularPredictor`)
- ⚖️ Dataset balancing using undersampling
- 🎯 Achieved 85%+ training accuracy
- 🌐 Interactive frontend built with plain HTML/CSS/JS
- 📦 Backend model and artifacts saved for deployment

---

## 📁 Project Structure

```
career-prediction/
├── backend/
│   ├── career_predictor.py
│   ├── improved_model_v2/
│   │   ├── models/
│   │   ├── fill_values.pkl
│   │   └── used_features.pkl
│   └── train_data1.csv
├── templates/
│   ├── index.html
├── static/
│   ├── style.css
└── README.md (this file)
```

---

## ⚙️ Tech Stack

### Frontend:
- HTML/CSS
- JavaScript

### Backend:
- Python
- pandas
- scikit-learn
- AutoGluon
- joblib

---

## 🧠 ML Model Details

- Target: `"What would you like to become when you grow up"`
- Preprocessing:
  - Null handling
  - Feature engineering: `Tech_Level` from `Tech-Savviness`
  - Data type conversion
  - Balancing classes via undersampling
- Model: `AutoGluon.TabularPredictor`
- Evaluation: Accuracy score and leaderboard

---

## 🧪 How to Run the Project

### 🔧 Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install required packages:
   ```bash
   pip install pandas scikit-learn autogluon joblib flask
   ```

3. Run the backend server:
   ```bash
   python app.py
   ```

4. Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📊 Input Features (Sample)

- Academic Performance (CGPA/Percentage)
- Tech-Savviness
- Daily Water Intake
- Number of Siblings
- Participation in Extracurricular Activities
- Risk-Taking Ability
- Leadership Experience
- Networking & Social Skills
- Previous Work Experience

---

## 🎯 Output

The model predicts the career goal, such as:
- Engineer
- Doctor
- Artist
- Researcher
- Entrepreneur
- And more...

---

## 🙌 Acknowledgments

- Project built during **HackMarch 2025** at **KLE Society**
- Thanks to all organizers and mentors for their support!

---

## 📄 License

This project is open-source and free to use for educational and research purposes.

---


