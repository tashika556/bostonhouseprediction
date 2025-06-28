import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score , mean_squared_error
import numpy as np

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
# Data Loading
def load_data():
    df = pd.read_csv("housing.csv")
    required_cols=['RM','LSTAT','PTRATIO','MEDV']
    assert all(col in df.columns for col in required_cols),"Missing required columns"
    df = clean_data(df)
    return df

df = load_data()

# Data Preprocessing
X = df.drop("MEDV",axis=1)
Y = df["MEDV"]

# Feature Scaling

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#Training model
@st.cache_resource
def train_model():
    model = RandomForestRegressor(
        n_estimators=200,  
        max_depth=10,     
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model()