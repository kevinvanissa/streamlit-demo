# Import Libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Setup
st.set_page_config("Diabetes Prediction Dashboard", layout="wide")
# Center title using Markdown and HTML
st.markdown("<h1 style='text-align: center;'>Diabetes Risk Explorer Demo</h1>", unsafe_allow_html=True)
#st.title("Diabetes Risk Explorer")
st.caption("Analyzing and predicting diabetes using the Pima Indians dataset.")

# Load Data with caching decorator
@st.cache_data
def load_data():
    url = "diabetes.csv"
    return pd.read_csv(url)

df = load_data()

# Score Cards Section
st.subheader("Score Cards")

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", f"{len(df)}")
col2.metric("Features", f"{df.shape[1] - 1}")
col3.metric("Diabetes Prevalence", f"{df['Outcome'].mean() * 100:.1f}%")

# Data Preview
with st.expander("Preview Dataset"):

    st.markdown("""
        This dataset contains medical diagnostic measurements of women from the Pima Indian heritage.
        Below is an ordered list explaining each feature.
    """)

    # Display ordered list with feature explanations
    st.subheader("Features Explanation")

    st.markdown("""
    1. **Pregnancies**: Number of times pregnant.
    2. **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
    3. **BloodPressure**: Diastolic blood pressure (mm Hg).
    4. **SkinThickness**: Triceps skin fold thickness (mm).
    5. **Insulin**: 2-Hour serum insulin (mu U/ml).
    6. **BMI**: Body mass index (weight in kg / (height in m)^2).
    7. **DiabetesPedigreeFunction**: A function which scores the likelihood of diabetes based on family history.
    8. **Age**: Age (years).
    9. **Outcome**: 1 if the person has diabetes, 0 otherwise.
    """)

    st.subheader("Raw Data")
    st.dataframe(df.head(), use_container_width=True)

# Interactive Plot
st.subheader("Explore Relationships")

x_axis = st.selectbox("X-axis", df.columns[:-1], index=1)
y_axis = st.selectbox("Y-axis", df.columns[:-1], index=5)
color_by = st.selectbox("Color by", ["Outcome"])

fig = px.scatter(
    df, x=x_axis, y=y_axis,
    color=df[color_by].map({0: "No Diabetes", 1: "Diabetes"}),
    labels={"color": "Diabetes Status"},
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
with st.expander("Correlation Matrix"):
    corr_fig = px.imshow(df.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(corr_fig, use_container_width=True)

# Model Training
st.subheader("Model Training & Evaluation")

@st.cache_resource
def train_model():
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, scaler, report

model, scaler, report = train_model()

# Model accuracy card
model_accuracy = report['accuracy']
st.metric("Model Accuracy", f"{model_accuracy * 100:.2f}%")

st.write("**Classification Report**")
report_df = pd.DataFrame(report).transpose().round(2)
st.dataframe(report_df, use_container_width=True)

# Live Prediction
st.subheader("Live Diabetes Risk Prediction")

st.markdown("Enter patient information to estimate diabetes risk:")

with st.form("diabetes_form"):
    cols = st.columns(4)

    pregnancies = cols[0].number_input("Pregnancies", 0, 20, 2)
    glucose = cols[1].number_input("Glucose", 0, 200, 100)
    bp = cols[2].number_input("Blood Pressure", 0, 140, 70)
    skin = cols[3].number_input("Skin Thickness", 0, 100, 20)

    insulin = cols[0].number_input("Insulin", 0.0, 900.0, 80.0)
    bmi = cols[1].number_input("BMI", 0.0, 80.0, 25.0)
    dpf = cols[2].number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = cols[3].number_input("Age", 10, 100, 30)

    predict_btn = st.form_submit_button("Predict")

    if predict_btn:
        input_df = pd.DataFrame([{
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }])

        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(f"Prediction: **Likely Diabetic** — Risk Score: **{prob*100:.2f}%**")
        else:
            st.success(f"Prediction: **Unlikely Diabetic** — Risk Score: **{(1 - prob)*100:.2f}%**")

# Footer
st.markdown("---")
st.caption("Dataset: Pima Indians Diabetes (UCI) | Demo Create by Kevin Miller")

