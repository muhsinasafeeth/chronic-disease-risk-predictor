import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ── Dark background ── */
    .stApp {
        background: #0D1117;
    }
    .block-container {
        padding: 1.5rem 2.5rem 3rem;
        max-width: 860px;
    }

    /* ── Hero header ── */
    .hero {
        background: linear-gradient(135deg, #161B22 0%, #1C2333 60%, #0D1117 100%);
        border: 1px solid #30363D;
        border-radius: 24px;
        padding: 2.8rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; left: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(46,134,171,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-icon { font-size: 3rem; margin-bottom: 0.75rem; }
    .hero h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #E6EDF3;
        margin: 0 0 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 0.9rem;
        color: #8B949E;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(46,134,171,0.15);
        border: 1px solid rgba(46,134,171,0.35);
        color: #58A6FF;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 3px 12px;
        border-radius: 20px;
        margin-top: 0.75rem;
    }

    /* ── Stats row ── */
    .stat-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: #2E86AB; }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #58A6FF;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #8B949E;
        margin-top: 3px;
        font-weight: 400;
    }

    /* ── Section cards ── */
    .card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.25rem;
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #58A6FF;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1.25rem;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid #21262D;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Sliders & inputs ── */
    .stSlider label { color: #C9D1D9 !important; font-size: 0.9rem !important; font-weight: 500 !important; }
    .stSlider > div > div > div { background: #2E86AB !important; }
    .stSelectbox label { color: #C9D1D9 !important; font-size: 0.9rem !important; font-weight: 500 !important; }
    div[data-baseweb="select"] > div {
        background: #21262D !important;
        border-color: #30363D !important;
        color: #C9D1D9 !important;
        border-radius: 10px !important;
    }

    /* ── Predict button ── */
    .stButton > button {
        background: linear-gradient(135deg, #2E86AB, #1a5f7a);
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.85rem 2rem;
        border-radius: 14px;
        border: none;
        width: 100%;
        letter-spacing: 0.4px;
        box-shadow: 0 4px 24px rgba(46,134,171,0.4);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(46,134,171,0.55);
    }

    /* ── Result cards ── */
    .result-normal {
        background: linear-gradient(135deg, #0D2818, #0F3320);
        border: 1px solid #2EA043;
        border-left: 5px solid #2EA043;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(46,160,67,0.12);
    }
    .result-prediab {
        background: linear-gradient(135deg, #271D0D, #2D2210);
        border: 1px solid #D29922;
        border-left: 5px solid #D29922;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(210,153,34,0.12);
    }
    .result-diabetic {
        background: linear-gradient(135deg, #2D0D0D, #330F0F);
        border: 1px solid #F85149;
        border-left: 5px solid #F85149;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(248,81,73,0.12);
    }
    .result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.5rem; }
    .result-subtitle { font-size: 0.95rem; opacity: 0.85; }

    /* ── Tips card ── */
    .tips-card {
        background: linear-gradient(135deg, #0D1B2A, #101C2E);
        border: 1px solid #1F3A5F;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.25rem;
    }
    .tip-item {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        color: #C9D1D9;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #21262D;
        margin: 1.5rem 0;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #1C1107;
        border: 1px solid #3D2B00;
        border-left: 4px solid #D29922;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #D29922;
        font-size: 0.82rem;
        line-height: 1.6;
        margin-top: 1rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #484F58;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1.25rem;
        border-top: 1px solid #21262D;
        line-height: 1.8;
    }

    /* ── Info pills ── */
    .pill {
        display: inline-block;
        background: #21262D;
        border: 1px solid #30363D;
        color: #8B949E;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 2px;
    }

    /* ── Section icon header ── */
    .section-icon { font-size: 1.1rem; }

    /* hide streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model  = joblib.load(os.path.join(base, '..', 'models', 'logistic_regression_model.pkl'))
    scaler = joblib.load(os.path.join(base, '..', 'models', 'scaler.pkl'))
    return model, scaler

model, scaler = load_model()

# ─────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🩺</div>
    <h1>Diabetes Risk Predictor</h1>
    <p>Enter your health measurements below to assess your diabetes risk using<br>
    a machine learning model trained on 6,041 real patients from CDC NHANES data.</p>
    <span class="hero-badge">⚕️ For Educational Purposes Only — Not a Medical Diagnosis</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# STATS ROW
# ─────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""<div class="stat-card">
        <div class="stat-value">6,041</div>
        <div class="stat-label">Patients Trained On</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""<div class="stat-card">
        <div class="stat-value">70%</div>
        <div class="stat-label">Diabetic Recall</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""<div class="stat-card">
        <div class="stat-value">3</div>
        <div class="stat-label">Risk Classes</div>
    </div>""", unsafe_allow_html=True)
with s4:
    st.markdown("""<div class="stat-card">
        <div class="stat-value">CDC</div>
        <div class="stat-label">Data Source</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FORM — PERSONAL INFO
# ─────────────────────────────────────────
st.markdown("""<div class="card">
    <div class="card-title"><span class="section-icon">👤</span> Personal Information</div>
</div>""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", 18, 80, 45)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        ethnicity = st.selectbox("Ethnicity", [
            "Non-Hispanic White", "Non-Hispanic Black",
            "Non-Hispanic Asian", "Mexican American",
            "Other Hispanic", "Other"
        ])

# ─────────────────────────────────────────
# FORM — BODY MEASUREMENTS
# ─────────────────────────────────────────
st.markdown("""<div class="card">
    <div class="card-title"><span class="section-icon">📏</span> Body Measurements</div>
</div>""", unsafe_allow_html=True)

with st.container():
    col4, col5, col6 = st.columns(3)
    with col4:
        bmi = st.slider("BMI (kg/m²)", 15.0, 60.0, 25.0, step=0.1)
    with col5:
        waist_cm = st.slider("Waist Circumference (cm)", 50.0, 160.0, 90.0, step=0.5)
    with col6:
        poverty_ratio = st.slider("Poverty Income Ratio", 0.0, 5.0, 2.5, step=0.1,
                                  help="Ratio of family income to poverty level. Higher = wealthier.")

# BMI indicator
if bmi < 18.5:
    bmi_note = "⚪ Underweight"
elif bmi < 25:
    bmi_note = "🟢 Normal weight"
elif bmi < 30:
    bmi_note = "🟡 Overweight"
else:
    bmi_note = "🔴 Obese"
st.caption(f"BMI Status: **{bmi_note}**")

# ─────────────────────────────────────────
# FORM — CLINICAL MEASUREMENTS
# ─────────────────────────────────────────
st.markdown("""<div class="card">
    <div class="card-title"><span class="section-icon">🩸</span> Clinical Measurements</div>
</div>""", unsafe_allow_html=True)

with st.container():
    col7, col8, col9 = st.columns(3)
    with col7:
        systolic_bp = st.slider("Systolic BP (mmHg)", 80, 200, 120,
                                help="Top number of blood pressure reading")
    with col8:
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 40, 120, 80,
                                 help="Bottom number of blood pressure reading")
    with col9:
        total_cholesterol = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)

# BP indicator
if systolic_bp < 120 and diastolic_bp < 80:
    bp_note = "🟢 Normal"
elif systolic_bp < 130:
    bp_note = "🟡 Elevated"
elif systolic_bp < 140:
    bp_note = "🟠 Hypertension Stage 1"
else:
    bp_note = "🔴 Hypertension Stage 2"
st.caption(f"Blood Pressure Status: **{bp_note}**")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
predict_clicked = st.button("🔍  Analyse My Diabetes Risk")

# ─────────────────────────────────────────
# FEATURE ENGINEERING FUNCTION
# ─────────────────────────────────────────
def build_features(age, gender, ethnicity, bmi, waist_cm,
                   poverty_ratio, systolic_bp, diastolic_bp, total_cholesterol):
    if bmi < 18.5:       bmi_cat = 'Underweight'
    elif bmi < 25.0:     bmi_cat = 'Normal'
    elif bmi < 30.0:     bmi_cat = 'Overweight'
    else:                bmi_cat = 'Obese'

    if age <= 30:        age_grp = '18-30'
    elif age <= 45:      age_grp = '31-45'
    elif age <= 60:      age_grp = '46-60'
    else:                age_grp = '60+'

    if systolic_bp < 120 and diastolic_bp < 80:    bp_cat = 'Normal'
    elif systolic_bp < 130 and diastolic_bp < 80:  bp_cat = 'Elevated'
    elif systolic_bp < 140 or diastolic_bp < 90:   bp_cat = 'Hypertension Stage 1'
    else:                                           bp_cat = 'Hypertension Stage 2'

    row = {
        'age': age, 'poverty_ratio': poverty_ratio,
        'bmi': bmi, 'waist_cm': waist_cm,
        'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'total_cholesterol': total_cholesterol,
        'gender_Female': 1 if gender == 'Female' else 0,
        'gender_Male'  : 1 if gender == 'Male'   else 0,
        'ethnicity_Mexican American'  : 1 if ethnicity == 'Mexican American'   else 0,
        'ethnicity_Non-Hispanic Asian': 1 if ethnicity == 'Non-Hispanic Asian'  else 0,
        'ethnicity_Non-Hispanic Black': 1 if ethnicity == 'Non-Hispanic Black'  else 0,
        'ethnicity_Non-Hispanic White': 1 if ethnicity == 'Non-Hispanic White'  else 0,
        'ethnicity_Other'             : 1 if ethnicity == 'Other'               else 0,
        'ethnicity_Other Hispanic'    : 1 if ethnicity == 'Other Hispanic'      else 0,
        'bmi_category_Normal'     : 1 if bmi_cat == 'Normal'      else 0,
        'bmi_category_Obese'      : 1 if bmi_cat == 'Obese'       else 0,
        'bmi_category_Overweight' : 1 if bmi_cat == 'Overweight'  else 0,
        'bmi_category_Underweight': 1 if bmi_cat == 'Underweight' else 0,
        'age_group_18-30': 1 if age_grp == '18-30' else 0,
        'age_group_31-45': 1 if age_grp == '31-45' else 0,
        'age_group_46-60': 1 if age_grp == '46-60' else 0,
        'age_group_60+'  : 1 if age_grp == '60+'   else 0,
        'bp_category_Elevated'            : 1 if bp_cat == 'Elevated'             else 0,
        'bp_category_Hypertension Stage 1': 1 if bp_cat == 'Hypertension Stage 1' else 0,
        'bp_category_Hypertension Stage 2': 1 if bp_cat == 'Hypertension Stage 2' else 0,
        'bp_category_Normal'              : 1 if bp_cat == 'Normal'               else 0,
    }
    return pd.DataFrame([row])

# ─────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────
if predict_clicked:

    input_df      = build_features(age, gender, ethnicity, bmi, waist_cm,
                                   poverty_ratio, systolic_bp, diastolic_bp, total_cholesterol)
    input_scaled  = scaler.transform(input_df)
    prediction    = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    label_map     = {0: 'Normal', 1: 'Pre-diabetic', 2: 'Diabetic'}
    result        = label_map[prediction]

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📋 Your Results")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result Card ──
    if prediction == 0:
        st.markdown(f"""
        <div class="result-normal">
            <div class="result-title" style="color:#3FB950;">🟢 {result}</div>
            <div class="result-subtitle" style="color:#7EE787;">
                Your risk indicators are within normal range.<br>
                Keep maintaining your healthy lifestyle!
            </div>
        </div>""", unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown(f"""
        <div class="result-prediab">
            <div class="result-title" style="color:#D29922;">🟡 {result}</div>
            <div class="result-subtitle" style="color:#E3B341;">
                Your indicators suggest elevated risk.<br>
                Early lifestyle changes can prevent progression to diabetes.
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-diabetic">
            <div class="result-title" style="color:#F85149;">🔴 {result}</div>
            <div class="result-subtitle" style="color:#FF7B72;">
                High risk indicators detected.<br>
                Please consult a healthcare professional as soon as possible.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability Chart ──
    st.markdown("""<div class="card">
        <div class="card-title"><span class="section-icon">📊</span> Risk Probability Breakdown</div>
    </div>""", unsafe_allow_html=True)

    fig = go.Figure()
    labels = ['Normal', 'Pre-diabetic', 'Diabetic']
    colors = ['#3FB950', '#D29922', '#F85149']
    bar_colors = ['rgba(63,185,80,0.85)', 'rgba(210,153,34,0.85)', 'rgba(248,81,73,0.85)']

    for i, (label, prob, color, bcolor) in enumerate(zip(labels, probabilities, colors, bar_colors)):
        fig.add_trace(go.Bar(
            x=[prob],
            y=[label],
            orientation='h',
            marker=dict(
                color=bcolor,
                line=dict(color=color, width=1.5)
            ),
            text=f'{prob:.1%}',
            textposition='outside',
            textfont=dict(color=color, size=14, family='Inter'),
            width=0.45,
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 1.25],
            showgrid=True,
            gridcolor='#21262D',
            tickformat='.0%',
            color='#8B949E',
            title=None
        ),
        yaxis=dict(
            showgrid=False,
            color='#C9D1D9',
            tickfont=dict(size=13)
        ),
        plot_bgcolor='#161B22',
        paper_bgcolor='#161B22',
        height=220,
        margin=dict(l=10, r=60, t=10, b=10),
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Gauge Chart ──
    risk_score = probabilities[1] * 0.5 + probabilities[2]
    gauge_color = '#3FB950' if risk_score < 0.3 else '#D29922' if risk_score < 0.6 else '#F85149'

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score * 100, 1),
        title={'text': "Overall Risk Score", 'font': {'color': '#C9D1D9', 'size': 14}},
        number={'suffix': '%', 'font': {'color': gauge_color, 'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8B949E', 'tickfont': {'color': '#8B949E'}},
            'bar': {'color': gauge_color, 'thickness': 0.25},
            'bgcolor': '#21262D',
            'bordercolor': '#30363D',
            'steps': [
                {'range': [0, 30],  'color': 'rgba(63,185,80,0.1)'},
                {'range': [30, 60], 'color': 'rgba(210,153,34,0.1)'},
                {'range': [60, 100],'color': 'rgba(248,81,73,0.1)'}
            ],
            'threshold': {
                'line': {'color': gauge_color, 'width': 3},
                'thickness': 0.75,
                'value': round(risk_score * 100, 1)
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='#161B22',
        font={'color': '#C9D1D9'},
        height=220,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Patient Summary ──
    st.markdown("""<div class="card">
        <div class="card-title"><span class="section-icon">🗂️</span> Your Health Summary</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Age", f"{age} yrs")
        st.metric("BMI", f"{bmi:.1f}")
    with c2:
        st.metric("Waist", f"{waist_cm:.0f} cm")
        st.metric("Systolic BP", f"{systolic_bp} mmHg")
    with c3:
        st.metric("Cholesterol", f"{total_cholesterol} mg/dL")
        st.metric("Diastolic BP", f"{diastolic_bp} mmHg")

    # ── Health Tips ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="tips-card">
        <div class="card-title" style="color:#58A6FF; border-bottom-color:#1F3A5F;">
            <span>💡</span> Personalised Health Recommendations
        </div>
    """, unsafe_allow_html=True)

    tips = []
    if bmi >= 30:
        tips.append(("⚖️", "Weight Management", "Your BMI indicates obesity. A 5–10% reduction in body weight can significantly lower diabetes risk."))
    elif bmi >= 25:
        tips.append(("⚖️", "Weight Management", "Your BMI indicates overweight. Small consistent changes in diet and activity can help."))
    if waist_cm > 102 and gender == 'Male':
        tips.append(("📏", "Abdominal Fat", "Waist above 102cm in men is a key diabetes risk marker. Focus on core exercises and reducing refined carbs."))
    if waist_cm > 88 and gender == 'Female':
        tips.append(("📏", "Abdominal Fat", "Waist above 88cm in women is a key diabetes risk marker. Aerobic activity helps reduce visceral fat."))
    if systolic_bp >= 130:
        tips.append(("❤️", "Blood Pressure", "Your blood pressure is elevated. Reduce salt intake, exercise regularly and limit alcohol."))
    if total_cholesterol > 240:
        tips.append(("🩸", "Cholesterol", "High cholesterol detected. Reduce saturated fats, eat more fibre and consider omega-3 rich foods."))
    if age >= 45:
        tips.append(("📅", "Age Risk", "Adults over 45 should get a diabetes screening test every 3 years regardless of symptoms."))
    if poverty_ratio < 1.5:
        tips.append(("🏥", "Healthcare Access", "Consider community health centres for affordable diabetes screening and preventive care."))

    tips.append(("🥗", "Nutrition", "Follow a balanced diet rich in vegetables, whole grains, legumes and lean proteins. Limit sugary drinks."))
    tips.append(("🏃", "Physical Activity", "Aim for at least 150 minutes of moderate exercise per week. Even 30 mins of walking daily helps."))

    for icon, title, desc in tips:
        st.markdown(f"""
        <div class="tip-item">
            <strong style="color:#C9D1D9;">{icon} {title}:</strong>
            <span style="color:#8B949E;"> {desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider for medical decisions.
        Model trained on NHANES 2017–2018 CDC survey data.
    </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div>
        <span class="pill">🏥 NHANES CDC Data</span>
        <span class="pill">🤖 Logistic Regression</span>
        <span class="pill">📊 6,041 Patients</span>
        <span class="pill">🐍 Python + Streamlit</span>
    </div>
    <br>
    Built as part of a Data Science Portfolio Project · 
    GitHub: <strong>muhsinasafeeth/chronic-disease-risk-predictor</strong>
    <br><br>
    © 2024 · For Educational Purposes Only · Not a Medical Diagnosis Tool
</div>
""", unsafe_allow_html=True)
