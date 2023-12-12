import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


s = pd.read_csv("social_media_usage.csv",)

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


ss = pd.DataFrame({
    "Linkedin":s["web1h"].apply(clean_sm),
    "income":np.where(s["income"]>9,np.nan, s["income"]),
    "education":np.where(s["educ2"]>8,np.nan,s["educ2"]),
    "parents":np.where(s["par"]== 1,1,0),
    "married":np.where(s["marital"]==1,1,0),
    "female":np.where(s["gender"]==2,1,0),
    "age": np.where(s["age"]>98, np.nan,s["age"])})

ss=ss.dropna()


y = ss["Linkedin"]
X = ss[['income', 'education', 'parents', 'married', 'female', 'age']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987)


lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)


import streamlit as st
st.title("LLinkedin Users Prediction Application")
st.subheader("Please configure demongrapchis to predict if someone will likely use Linkedin")

income= st.selectbox("Pick one",["less than $10,000", "10 to under $20,000", "20 to under $30,000", "30 to under $40,000", "40 to under $50,000", "50 to under $75,000","75 to under $100,000", "100 to under $150,000", "150,00 or more", "Don't know", "Refused"])
