import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cloudpickle
from pathlib import Path

# Path to the folder containing the .py file
APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "model_pipeline.pkl"
THRESH_PATH = APP_DIR / "best_threshold.pkl"

# Caching to improve app speed performance
@st.cache_resource
def load_pipeline():
    with MODEL_PATH.open("rb") as f:
        return cloudpickle.load(f)
    
@st.cache_data
def load_threshold():
    return joblib.load(THRESH_PATH)

# Load pipeline and threshold
pipeline = load_pipeline()
best_threshold = load_threshold()

# Title
st.markdown("# :credit_card: Credit Card Churn Prediction")

mode = st.radio("Select Prediction Mode", ["Single Customer Input", "Batch Upload (CSV)"])

# --- Batch Upload ---
if mode == "Batch Upload (CSV)":
    st.info("Upload a CSV file with original customer features")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Predict probabilities
        probs = pipeline.predict_proba(df)[:, 1]
        preds = (probs >= best_threshold).astype(int)

        # Output
        df_output = df.copy()
        df_output['Churn Probability'] = np.round(probs, 3)
        df_output['Prediction'] = np.where(preds == 1, 'Churn', 'Stay')

        st.success("Prediction complete!")
        st.write(df_output)

        # Download results
        csv = df_output.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", data=csv, file_name="churn_predictions.csv", mime="text/csv")

# --- Single Input ---
if mode == "Single Customer Input":
    with st.form("input_form"):
        Customer_Age = st.slider("Age", 18, 90, 35)
        Gender = st.selectbox("Gender", ['Female', 'Male'])
        Dependent_count = st.slider("Number of Dependents", 0, 10, 2)
        Education_Level = st.selectbox("Education Level", ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', 'Unknown'])
        Marital_Status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unknown'])
        Income_Category = st.selectbox("Income Category", ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown'])
        Card_Category = st.selectbox("Card Type", ['Blue', 'Silver', 'Gold', 'Platinum'])
        Months_on_book = st.slider("Months on Book", 12, 60, 36)
        Total_Relationship_Count = st.slider("Total Relationship Count", 1, 10, 3)
        Months_Inactive_12_mon = st.slider("Inactive Months (Last 12)", 0, 12, 2)
        Contacts_Count_12_mon = st.slider("Contacts (Last 12)", 0, 20, 3)
        Credit_Limit = st.number_input("Credit Limit", 100.0, 100000.0, 5000.0)
        Total_Revolving_Bal = st.number_input("Revolving Balance", 0.0, 50000.0, 1000.0)
        Avg_Open_To_Buy = st.number_input("Avg Open To Buy", 0.0, 100000.0, 3000.0)
        Total_Amt_Chng_Q4_Q1 = st.number_input("Transaction Amount Change Q4/Q1", 0.0, 5.0, 1.0, 0.001, format="%.3f")
        Total_Trans_Amt = st.number_input("Total Transaction Amount", 0.0, 20000.0, 5000.0)
        Total_Trans_Ct = st.slider("Total Transaction Count", 0, 300, 80)
        Total_Ct_Chng_Q4_Q1 = st.number_input("Transaction Count Change Q4/Q1", 0.0, 5.0, 1.0, 0.001, format="%.3f")

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([{
                'Customer_Age': Customer_Age,
                'Gender': Gender,
                'Dependent_count': Dependent_count,
                'Education_Level': Education_Level,
                'Marital_Status': Marital_Status,
                'Income_Category': Income_Category,
                'Card_Category': Card_Category,
                'Months_on_book': Months_on_book,
                'Total_Relationship_Count': Total_Relationship_Count,
                'Months_Inactive_12_mon': Months_Inactive_12_mon,
                'Contacts_Count_12_mon': Contacts_Count_12_mon,
                'Credit_Limit': Credit_Limit,
                'Total_Revolving_Bal': Total_Revolving_Bal,
                'Avg_Open_To_Buy': Avg_Open_To_Buy,
                'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1,
                'Total_Trans_Amt': Total_Trans_Amt,
                'Total_Trans_Ct': Total_Trans_Ct,
                'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1
            }])

            prob = pipeline.predict_proba(input_df)[0][1]
            pred = 1 if prob >= best_threshold else 0

            st.subheader("Prediction")
            color = 'red' if pred == 1 else 'green'
            label = 'churn' if pred == 1 else 'stay'
            st.markdown(f"Churn Probability: <span style='color:{color}; font-weight:bold;'>{prob:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"Prediction: Customer is likely to <span style='color:{color}; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)