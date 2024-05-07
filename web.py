import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import requests
def send_post_request(api_url, json_data):
    response = requests.post(api_url, json=json_data)
    if response.status_code == 200:
        return response.json()
    else:
        return None
def show_predict_pag():
    st.write("Software Developer Stroke Prediction")
    api_url = "https://machine-api-eq7w.onrender.com/predict"


    out = {}
    out['age']=st.slider("Age", 0, 100, 50)

    out['hypertension'] = 1 if st.radio("Hypertension", ["Yes", "NO"]) == "Yes" else 0
    out['heart_disease'] = 1 if st.radio("Heart Disease", ["Yes", "NO"]) == "Yes" else 0

    out['gender']=st.selectbox("Gender", ["Male", "Female"])

    out['ever_married']=st.radio("Ever Married", ["Yes", "NO"])

    work_type = ["children", "Self-employed", "Private", "Govt_job", "Never_worked"]

    out['work_type']=st.selectbox("Work Type", work_type)
    out['Residence_type']=st.radio("Residence Type", ["Urban", "Rural"])


    out['avg_glucose_level']=st.slider("Avg Glucose Level", 50, 300, 150)       

    out['bmi']=st.slider("BMI", 10, 100, 25)            
    smoking_status = ["formerly smoked", "never smoked", "smokes"]          
    out['smoking_status']=st.selectbox("Smoking Status", smoking_status)
    out['smoking_not_found'] = "True" if st.radio("Smoking Not Found", ["Yes", "NO"]) == "Yes" else "False"
    ok = st.button("Predict")
    data={
        "gender": out['gender'],
        "age": out['age'],
        "hypertension": out['hypertension'],
        "heart_disease": out['heart_disease'],
        "ever_married": out['ever_married'],
        "work_type": out['work_type'],
        "Residence_type": out['Residence_type'],
        "avg_glucose_level": out['avg_glucose_level'],
        "bmi": out['bmi'],
        "smoking_status": out['smoking_status'],
        "smoking_not_found": out['smoking_not_found']
    }
    if ok:
        response = send_post_request(api_url, data)
        st.write("The predict: " + ("Potential Stroke" if response['prediction']  else "Clear"))



def show_explore_pag():
    df = pd.read_csv("dataset.csv")
    num_cols = ["bmi", "avg_glucose_level", "age"]
    cat_cols = list(df.drop(columns=num_cols + ["stroke"]).columns)
    target_col = "stroke"

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    st.write("### Histogram for Stroke Numerical Columns")
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, ax=ax[i], hue=target_col, kde=True, bins=50)
    st.pyplot(fig)

    st.write("### Countplot for Stroke Output")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="stroke", ax=ax)
    st.pyplot(fig)

    st.write("### Countplot for Smoking Status")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="smoking_status", ax=ax)
    st.pyplot(fig)

    st.write("### Countplot for Work Type")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="work_type", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 
    st.pyplot(fig)        
pag=st.sidebar.selectbox("Explore Or Prediction", ("Predict", "Explore"))
if pag == "Predict":

   show_predict_pag()
else:

    show_explore_pag()
