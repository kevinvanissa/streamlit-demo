# Import Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Set Streamlit Title etc 
st.set_page_config(page_title="Medical Cost Dashboard", layout="wide")
st.title("Medical Cost Analysis & Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# Sidebar: show raw data
if st.sidebar.checkbox("Show Raw Data"):

    # Feature Explanation
    st.subheader("Column Definitions")
    st.markdown("""
    1. **age**: Age of the individual (in years).
    2. **sex**: Gender of the individual (male or female).
    3. **bmi**: Body mass index (weight in kg / (height in m)^2).
    4. **children**: Number of children/dependents covered by the health insurance.
    5. **smoker**: Whether the individual is a smoker (yes or no).
    6. **region**: Geographical region of the individual (northeast, northwest, southeast, southwest).
    7. **charges**: The medical charges billed to the individual (in USD).
    """)

    st.subheader("Raw Data")
    st.dataframe(df.head(20))

# Encode categorical variables
def preprocess(df):
    df_copy = df.copy()
    le = LabelEncoder()
    for col in ['sex', 'smoker', 'region']:
        df_copy[col] = le.fit_transform(df_copy[col])
    return df_copy

df_encoded = preprocess(df)

# Section 1: Dataset Overview
st.header("Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Average Cost", f"${df['charges'].mean():,.2f}")
col3.metric("Smokers %", f"{(df['smoker'] == 'yes').mean()*100:.2f}%")

# Section 2: Visualizations
st.subheader("Feature Insights")

tab1, tab2, tab3 = st.tabs(["Distribution", "Correlations", "Boxplots"])

with tab1:
    st.write("Age and BMI Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['age'], kde=True, ax=ax[0], color="skyblue")
    ax[0].set_title("Age Distribution")
    sns.histplot(df['bmi'], kde=True, ax=ax[1], color="salmon")
    ax[1].set_title("BMI Distribution")
    st.pyplot(fig)

with tab2:
    st.write("Correlation Heatmap")
    corr = df_encoded.corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab3:
    st.write("Medical Charges by Smoker and Region")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="smoker", y="charges", data=df, ax=ax[0], hue="smoker", legend=False)
    ax[0].set_title("Smoker vs Charges")
    #sns.boxplot(x="region", y="charges", data=df, ax=ax[1], palette="Set3")
    sns.boxplot(x="region", y="charges", data=df, ax=ax[1], hue="region", legend=False)
    ax[1].set_title("Region vs Charges")
    st.pyplot(fig)

# Section 3: Model Training
st.header("Train Cost Prediction Model")

@st.cache_resource
def train_model():
    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, mse

model, mse = train_model()
st.success(f"Model trained! Test MSE: ${mse:,.2f}")

# Section 4: Live Prediction
st.header("Predict Your Medical Cost")

with st.form("prediction_form"):
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 1)
    smoker = st.selectbox("Smoker?", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    submitted = st.form_submit_button("Predict Medical Cost $$")

    if submitted:
        input_df = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == "male" else 0],
            'bmi': [bmi],
            'children': [children],
            'smoker': [1 if smoker == "yes" else 0],
            'region': [ {"northeast":0, "northwest":1, "southeast":2, "southwest":3}[region] ]
        })

        prediction = model.predict(input_df)[0]
        st.subheader(f"Estimated Cost: **${prediction:,.2f}**")

# Footer
st.markdown("---")
st.markdown("Insurance Predictor | By Kevin Miller")


