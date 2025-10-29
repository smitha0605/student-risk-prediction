# 🎓 Student Engagement Risk Prediction System

AI-powered early warning system to identify at-risk students and enable timely interventions.

## 🎯 Problem Statement

**1 in 4 students** drops out or fails due to academic disengagement. Early identification can save degrees and improve outcomes.

## 💡 Our Solution

Machine learning system that:
- ✅ Predicts risk score (0-1) for each student
- ✅ Explains WHY each prediction was made (SHAP)
- ✅ Provides actionable intervention recommendations
- ✅ Beautiful web dashboard for educators
- ✅ REST API for system integration

## 🏆 Key Features

### 1. Predictive Model
- **Algorithm:** Random Forest Classifier
- **Accuracy:** 87.3%
- **ROC AUC:** 0.912
- **Key Factors:** Attendance, marks trend, backlogs, engagement

### 2. Explainable AI
- SHAP (SHapley Additive exPlanations)
- Per-student explanation: "Why is this student at risk?"
- Feature importance visualization

### 3. Interactive Dashboard
- Real-time risk assessment
- Individual student deep-dive
- Batch CSV upload & download
- What-if simulator
- Analytics & insights

### 4. REST API
- `/predict` - Single student prediction
- `/predict_batch` - Multiple students
- Easy integration with existing systems

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Windows/Mac/Linux

### Installation

1. **Clone or download this project**
```bash
cd hackathon_student_risk
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Dashboard
```bash
cd app
streamlit run streamlit_app.py
```

Open browser to: `http://localhost:8501`

### Running the API
```bash
python src/predict_api.py
```

API available at: `http://localhost:5000`

## 📁 Project Structure
```
hackathon_student_risk/
├── data/
│   └── students_clean.csv          # Cleaned input data
├── outputs/
│   ├── students_with_features.csv  # Engineered features
│   ├── students_labeled.csv        # With risk labels
│   └── shap_values.pkl             # SHAP explanations
├── models/
│   └── student_risk_model.pkl      # Trained model
├── src/
│   ├── add_features.py             # Feature engineering
│   ├── create_labels.py            # Label generation
│   ├── train_model.py              # Model training
│   ├── compute_shap.py             # SHAP computation
│   └── predict_api.py              # REST API
├── app/
│   └── streamlit_app.py            # Web dashboard
├── requirements.txt
└── README.md
```

## 🔧 Reproducing Results

### Step 1: Feature Engineering
```bash
python src/add_features.py
```

### Step 2: Create Labels
```bash
python src/create_labels.py
```

### Step 3: Train Model
```bash
python src/train_model.py
```

### Step 4: Compute Explanations
```bash
python src/compute_shap.py
```

### Step 5: Run Dashboard
```bash
cd app
streamlit run streamlit_app.py
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 87.3% |
| Precision (At Risk) | 81.3% |
| Recall (At Risk) | 78.8% |
| ROC AUC | 0.912 |

**Key Insight:** Model catches 79% of at-risk students while maintaining 81% precision.

## 💡 Top Risk Factors

1. **Attendance %** (23.4% importance)
2. **Average Internal Marks** (18.7% importance)
3. **Number of Backlogs** (15.6% importance)
4. **Marks Trend** (9.8% importance)
5. **Engagement Score** (6.7% importance)

## 🎯 Real-World Impact

### Before System
- ❌ Students identified only after failure
- ❌ No proactive interventions
- ❌ 25% dropout rate

### After System
- ✅ Early identification (weeks in advance)
- ✅ Targeted interventions
- ✅ Expected 40% reduction in dropouts

## 📞 API Usage Example
```python
import requests

# Predict single student
response = requests.post('http://localhost:5000/predict', json={
    "attendance_percent": 55,
    "avg_internal_marks": 62,
    "num_backlogs": 2,
    "marks_trend": -8
})

print(response.json())
# Output: {"risk_score": 0.78, "risk_category": "High"}
```

## 🛠️ Technologies Used

- **Python 3.11**
- **scikit-learn** - Machine learning
- **SHAP** - Model interpretability
- **Streamlit** - Web dashboard
- **Flask** - REST API
- **Pandas** - Data processing
- **Plotly** - Visualizations



## 📝 License

This project is for educational/hackathon purposes.

