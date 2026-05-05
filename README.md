# 🏥 Chronic Disease Risk Predictor

> Predicting Diabetes / Pre-diabetes Risk using real-world CDC NHANES Data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chronicdata-analytics.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

---

## 🌐 Live Demo
👉 **[https://chronicdata-analytics.streamlit.app/](https://chronicdata-analytics.streamlit.app/)**

---

## 📌 Project Overview

This project builds an end-to-end machine learning pipeline to predict 
whether a patient is **Normal**, **Pre-diabetic**, or **Diabetic** based 
on real health measurements.

Unlike typical portfolio projects, this uses **raw, uncleaned government 
survey data** from the CDC NHANES 2017-2018 dataset — the same data used 
by real epidemiologists and medical researchers.

---

## 🗂️ Dataset

**Source:** [CDC NHANES 2017-2018](https://wwwn.cdc.gov/nchs/nhanes/)

| File | Contents |
|---|---|
| `DEMO_J.XPT` | Demographics — age, gender, ethnicity, income |
| `BMX_J.XPT` | Body measurements — BMI, waist circumference |
| `GHB_J.XPT` | HbA1c lab results — target variable |
| `DIQ_J.XPT` | Self-reported diabetes questionnaire |
| `BPX_J.XPT` | Blood pressure measurements |
| `TCHOL_J.XPT` | Total cholesterol levels |

Raw `.XPT` (SAS format) files were downloaded directly from CDC — 
not pre-cleaned Kaggle datasets.

---

## 🔬 Methodology

### 1. Data Acquisition
- Downloaded 6 raw XPT files from CDC NHANES
- Loaded using `pandas.read_sas()`

### 2. Exploratory Data Analysis
- Identified class imbalance — Normal 62%, Pre-diabetic 26%, Diabetic 12%
- Found coded values (e.g. `9.0` = "refused to answer")
- Discovered missing data patterns across all 6 tables

### 3. Data Cleaning
- Decoded all NHANES coded variables
- Applied **group-wise smart imputation:**
  - `poverty_ratio` → median by ethnicity
  - `bmi`, `waist_cm` → median by gender
  - `systolic_bp`, `diastolic_bp` → median by age group
  - `total_cholesterol` → overall median
  - `hba1c` → dropped (never impute target variable!)
- Merged all 6 tables on `SEQN` participant ID
- Final clean dataset: **6,041 patients**

### 4. Feature Engineering
- BMI categories (WHO standard)
- Age groups (clinical brackets)
- Blood pressure categories (AHA standard)
- One-hot encoding of all categorical variables
- Final feature set: **28 features**

### 5. Modeling
Three models trained and compared:

| Model | F1 (macro) | Recall (macro) | Diabetic Recall |
|---|---|---|---|
| **Logistic Regression** | **0.502** | **0.561** | **70%** 🥇 |
| XGBoost | 0.498 | 0.511 | 41% |
| Random Forest | 0.445 | 0.442 | 13% |

**Logistic Regression chosen** because on imbalanced medical data, 
higher recall on the diabetic minority class is more clinically 
valuable than overall accuracy.

### 6. Deployment
- Built with **Streamlit** — clean dark-themed UI
- Features: sliders, probability chart, gauge chart, personalised health tips
- Deployed on **Streamlit Cloud** — publicly accessible

---

## 📁 Project Structure
chronic-disease-risk-predictor/
│
├── data/
│   ├── raw/          ← Raw CDC NHANES XPT files (not pushed to GitHub)
│   └── processed/    ← Cleaned CSV files (not pushed to GitHub)
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_modeling.ipynb
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── scaler.pkl
│
├── app/
│   └── app.py              ← Streamlit web application
│
├── reports/
│   ├── diabetes_distribution.png
│   ├── feature_correlation.png
│   └── confusion_matrices.png
│
├── requirements.txt
└── README.md
---

## 🧠 Key Learnings

- **Real data is messy** — 6 tables, coded columns, refused answers
- **Never impute the target variable** — corrupts labels
- **Group-wise imputation beats overall median** — clinical context matters
- **Simpler models can win** — LR beat XGBoost on diabetic recall
- **Recall over accuracy** — missing a diabetic is clinically dangerous
- **Class imbalance is real** — required `class_weight='balanced'`
- **Age is the strongest predictor** — correlation of 0.46 with diabetes

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| Scikit-learn | ML modeling |
| XGBoost | Gradient boosting |
| Plotly | Interactive charts |
| Streamlit | Web deployment |
| GitHub | Version control |

---

## ▶️ Run Locally

```bash
git clone https://github.com/muhsinasafeeth/chronic-disease-risk-predictor.git
cd chronic-disease-risk-predictor
pip install -r requirements.txt
streamlit run app/app.py
```

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and is NOT a substitute 
for professional medical advice, diagnosis, or treatment. Always consult 
a qualified healthcare provider for medical decisions.

---

## 👨‍💻 Author

**Muhsin Asafeeth**
GitHub: [@muhsinasafeeth](https://github.com/muhsinasafeeth)
