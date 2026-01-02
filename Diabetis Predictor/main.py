import streamlit as st
import pandas as pd
import kagglehub
import numpy as np
from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setting up my app layout
# I'm giving my browser tab a name and an icon so it looks professional.
st.set_page_config(page_title="🏥Diabetes AI Checker", page_icon="🏥", layout="wide")


# 2. Building the "Brain" of my app
# I'm using @st.cache_resource so my app "remembers" the AI model.
# This way, it doesn't have to relearn everything every time I click a button.
@st.cache_resource
def train_medical_model():
    # I'm pulling the real-world medical data directly from Kaggle's servers.
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "uciml/pima-indians-diabetes-database",
        "diabetes.csv",
    )

    # DATA CLEANING:
    # I noticed some patients have 0 for Blood Pressure or BMI, which is impossible.
    # I'm replacing those "fake" 0s with the average (mean) of the rest of the group
    # so my AI doesn't get distracted by bad data.
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].mean())

    # I'm splitting my data: 'X' is the patient's stats, and 'y' is the answer (Diabetes or not).
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # I'm saving 20% of the data for a "Final Exam" to test how accurate my AI is.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # I'm choosing a 'Random Forest' model.
    # It’s basically a team of 100 mini-decision trees that vote on the final result.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # I'm calculating my accuracy score here.
    accuracy = model.score(X_test, y_test)
    return model, X.columns, accuracy


# 3. Running the training
# I'm showing a spinner so the user knows the AI is "thinking" and loading the data.
with st.spinner("🤖 My AI is reviewing historical medical records..."):
    model, feature_names, model_accuracy = train_medical_model()

# 4. Designing my Dashboard
st.title("🏥Diabetes Health Tracker")
# I'm displaying my model's accuracy right at the top so users can see how reliable it is.
st.markdown(f"**System Status:** Model Active | **My AI's Accuracy:** {model_accuracy:.1%}")
st.write("I built this tool to analyze patient vitals and predict diabetes risk. I got inspired by my dad(a doctor) who's helped diabetic patients in the past and taught me how important tracking diabetis parameters is important to the health of the individual patient and the well being of the community.")
st.markdown("---")

# I'm creating 3 columns to keep my sliders organized and easy to read.
col_a, col_b, col_c = st.columns(3)

with col_a:
    preg = st.number_input("Pregnancies", 0, 17, 3)
    glu = st.slider("Glucose Level (mg/dL)", 0, 200, 117)
    bp = st.slider("Blood Pressure (mm Hg)", 0, 122, 72)

with col_b:
    skin = st.slider("Skin Thickness (mm)", 0, 99, 23)
    ins = st.slider("Insulin (mu U/ml)", 0, 846, 30)
    bmi = st.slider("BMI (Body Mass Index)", 0.0, 67.0, 32.0)

with col_c:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.4, 0.37, help="Genetic risk score")
    age = st.number_input("Patient Age", 21, 81, 29)

# I'm taking all the slider values and putting them into a table the AI can read.
user_input = pd.DataFrame([[preg, glu, bp, skin, ins, bmi, dpf, age]], columns=feature_names)

# 5. Showing the Result
st.markdown("---")
if st.button("🚀 Run My Diagnostic Analysis"):
    # I'm asking the model to predict the outcome and the probability percentage.
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)

    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction[0] == 1:
            st.error("### ⚠️ HIGH RISK DETECTED")
            st.write("My AI suggests this profile matches high-risk diabetes patterns.")
        else:
            st.success("### ✅ LOW RISK DETECTED")
            st.write("My AI suggests these vitals are currently in a low-risk range.")

    with res_col2:
        # I'm showing the exact percentage of risk the AI calculated.
        risk_score = round(probability[0][1] * 100, 1)
        st.metric("Risk Probability", f"{risk_score}%")

    # 6. Explaining the "Why"
    # I'm adding a chart to show which factor (like Glucose) influenced my AI's decision the most.
    st.markdown("---")
    st.subheader("🔬 Factors My AI Looked At")
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    feat_importances.plot(kind='barh', color='skyblue', ax=ax)
    st.pyplot(fig)

# 7. Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.write("**Model Type:** Random Forest")
st.sidebar.info("Disclaimer: I built this for my portfolio plus through the influence my dad has on me. Please see a doctor for real medical advice!")