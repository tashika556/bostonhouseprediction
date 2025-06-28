import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FRIENDLY_NAMES = {
    "RM" : "Average Rooms",
    "LSTAT" : "Poverty Rate (%)",
    "PTRATIO" : "School Quality (Student:Teacher Ratio)"
}

FEATURE_DESCRIPTION= {
    "RM" : "Average number of rooms in home",
    "LSTAT" : "Percentage of lower-status population",
    "PTRATIO" : "Student to teacher ratio in nearby schools"
}

@st.cache_data

def load_data():
    df = pd.read_csv("housing.csv")
    required_cols=['RM','LSTAT','PTRATIO','MEDV']
    assert all(col in df.columns for col in required_cols),"Missing required columns"
    df = clean_data(df)
    return df

df = load_data()